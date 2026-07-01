#!/usr/bin/env python3
"""Run v78 TTT long-window regime-action smokes and evaluate window5 Sim(3).

The tool is diagnostic-only.  It runs one contiguous multi-chunk KITTI window,
compares existing no-GT TTT commit protection hooks, and writes an explicit
decision file that does not claim v78 success.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_v78_bad_window_tables import _evaluate_run  # noqa: E402


DEFAULT_DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset")
DEFAULT_CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
DEFAULT_CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
DEFAULT_CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti02_hmc_v2/acl2_v78tf_pca_grounded_memory_control/report_final/"
    "phase4_ttt_long_window_regime_action_smoke_v1"
)
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


CASES: dict[str, dict[str, Any]] = {
    "LW0_READPATH_NATIVE": {
        "description": "read-path native control; no TTT semantic write action",
        "hybrid_memory_mode": "read_path_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
        "semantic_role_control_mode": "none",
        "extra": [],
    },
    "LW1_TTT_SEMANTIC_BASE": {
        "description": "TTT semantic write baseline with no commit protection hook",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [],
    },
    "LW2_TTT_TAIL_CONTINUITY_COMMIT_GUARD": {
        "description": "no-GT tail-state continuity guard; protects commits when candidate/native state continuity looks risky",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_commit_filter_mode",
            "tail_state_continuity_selective_commit",
            "--ttt_write_commit_filter_risk_source",
            "d_tok",
            "--ttt_write_commit_filter_scope",
            "both_overlap",
            "--ttt_write_commit_filter_stat",
            "mean",
            "--ttt_write_commit_filter_min",
            "0.25",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "all",
        ],
    },
    "LW3_TTT_DTOK_OLD_DECAY_Q90": {
        "description": "no-GT high-D_tok overlap commit decay; protects previous TTT state in risky overlap",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_commit_filter_mode",
            "old_decay_by_risk",
            "--ttt_write_commit_filter_risk_source",
            "d_tok",
            "--ttt_write_commit_filter_scope",
            "both_overlap",
            "--ttt_write_commit_filter_stat",
            "q90",
            "--ttt_write_commit_filter_base",
            "1.0",
            "--ttt_write_commit_filter_gain",
            "0.85",
            "--ttt_write_commit_filter_min",
            "0.15",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "all",
        ],
    },
    "LW4_TTT_COMMIT_EMA_050": {
        "description": "no-GT conservative TTT commit EMA; global protection control",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_commit_ema_alpha",
            "0.50",
            "--ttt_write_commit_ema_branch_mask",
            "all",
        ],
    },
    "LW5_TTT_FREEZE_ALL_DIAGNOSTIC": {
        "description": "diagnostic-only full TTT commit freeze; tests whether long-window TTT persistence is harmful",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [],
        "freeze_all": True,
        "diagnostic_control": True,
    },
    "LW6_TTT_KV_FRAME_STATIC_B0": {
        "description": "branch0 K/V frame-static replay feature gate; tests no-GT stable-corridor anchor protection",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_feature_gate_mode",
            "kv_frame_static_center",
            "--ttt_write_replay_feature_gate_rho",
            "0.35",
            "--ttt_write_replay_feature_gate_min",
            "0.65",
            "--ttt_write_replay_feature_gate_branch_mask",
            "0",
        ],
    },
    "LW7_TTT_NATIVE_ORTHO_SUPPRESS_B0": {
        "description": "branch0 native-delta orthogonal suppression; keeps semantic TTT correction aligned to native TTT continuity",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_native_delta_gate_mode",
            "orthogonal_suppress",
            "--ttt_write_native_delta_gate_fallback",
            "0.25",
            "--ttt_write_native_delta_gate_branch_mask",
            "0",
        ],
    },
    "LW8_TTT_KV_STATIC_PLUS_ORTHO_B0": {
        "description": "branch0 combined K/V frame-static gate plus native orthogonal suppression",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
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
    "LW9_TTT_KV_FRAME_STATIC_WEAK_B0": {
        "description": "weak branch0 K/V frame-static replay feature gate; lower-rho audit of stable-corridor anchor protection",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_feature_gate_mode",
            "kv_frame_static_center",
            "--ttt_write_replay_feature_gate_rho",
            "0.10",
            "--ttt_write_replay_feature_gate_min",
            "0.90",
            "--ttt_write_replay_feature_gate_branch_mask",
            "0",
        ],
    },
    "LW10_TTT_V_FRAME_STATIC_WEAK_B0": {
        "description": "weak branch0 V-only frame-static replay feature gate; tests value alignment without key-direction distortion",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_feature_gate_mode",
            "v_frame_static_center",
            "--ttt_write_replay_feature_gate_rho",
            "0.10",
            "--ttt_write_replay_feature_gate_min",
            "0.90",
            "--ttt_write_replay_feature_gate_branch_mask",
            "0",
        ],
    },
    "LW11_TTT_STATE_ENERGY_DIR_B0_MIN075": {
        "description": "branch0 state-energy directional commit guard; no chunk ids, default candidate/native delta geometry, min alpha 0.75",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_commit_filter_mode",
            "state_energy_directional_commit",
            "--ttt_write_commit_filter_min",
            "0.75",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "0",
        ],
    },
    "LW13_TTT_TAIL_STATE_SELECTIVE_B0_MIN075": {
        "description": "branch0 tail-risk selective state/delta commit guard; no chunk ids, min alpha 0.75",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_commit_filter_mode",
            "tail_state_selective_commit",
            "--ttt_write_commit_filter_min",
            "0.75",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "0",
        ],
    },
    "LW14_TTT_TAIL_SOFT_DIR_B0_MIN075": {
        "description": "branch0 tail-risk soft directional commit guard; risk-gated state/delta damping, min alpha 0.75",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_commit_filter_mode",
            "tail_state_soft_directional_commit",
            "--ttt_write_commit_filter_min",
            "0.75",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "0",
        ],
    },
    "LW12_TTT_OVERLAP_DYNAMIC_VETO_B0_BLEND050": {
        "description": "branch0 scoped dynamic-veto replay token filter over both overlaps; blend 0.50 to test selective overlap protection",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "0.55",
            "--ttt_write_replay_token_filter_branch_mask",
            "0",
            "--ttt_write_replay_token_filter_blend",
            "0.50",
            "--ttt_write_replay_token_filter_blend_mode",
            "linear",
        ],
    },
    "LW15_TTT_OVERLAP_DYNAMIC_TTL_B0_SUB100": {
        "description": "branch0 TTT6 one-hop dynamic residual TTL over both overlaps; stores filtered dynamic residual and subtracts it on the next commit",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "0.55",
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
    "LW16_TTT_OVERLAP_DYNAMIC_TTL_RANDOM_ROLE_B0_SUB100": {
        "description": "random same-mass role control for LW15 one-hop dynamic residual TTL",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79016",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "0.55",
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
    "LW17_TTT_OVERLAP_ALIGNED_DYNAMIC_TTL_B0_BLEND050_SUB100": {
        "description": "branch0 TTT6 aligned-dynamic TTL over both overlaps; keeps dynamic residual only when aligned to the static filtered update and subtracts it on next commit",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "0.55",
            "--ttt_write_replay_token_filter_branch_mask",
            "0",
            "--ttt_write_replay_token_filter_blend",
            "0.50",
            "--ttt_write_replay_token_filter_blend_mode",
            "ttl_aligned_dynamic",
            "--ttt_write_transient_delta_subtract_scale",
            "1.0",
            "--ttt_write_transient_delta_branch_mask",
            "0",
            "--ttt_write_transient_delta_ttl",
            "1",
        ],
    },
    "LW18_TTT_OVERLAP_ALIGNED_DYNAMIC_TTL_RANDOM_ROLE_B0_BLEND050_SUB100": {
        "description": "random same-mass role control for LW17 aligned-dynamic TTL",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79018",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "0.55",
            "--ttt_write_replay_token_filter_branch_mask",
            "0",
            "--ttt_write_replay_token_filter_blend",
            "0.50",
            "--ttt_write_replay_token_filter_blend_mode",
            "ttl_aligned_dynamic",
            "--ttt_write_transient_delta_subtract_scale",
            "1.0",
            "--ttt_write_transient_delta_branch_mask",
            "0",
            "--ttt_write_transient_delta_ttl",
            "1",
        ],
    },
    "LW19_TTT_OVERLAP_DYNAMIC_TTL_B0_PRIOR100": {
        "description": "branch0 one-hop dynamic TTL with observed-prior median threshold 1.00 over both overlaps",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
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
    "LW20_TTT_OVERLAP_DYNAMIC_TTL_RANDOM_ROLE_B0_PRIOR100": {
        "description": "random same-mass role control for LW19 observed-prior median threshold TTL",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79020",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
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
    "LW33_TTT_OVERLAP_DYNAMIC_FILTERONLY_B0_PRIOR100": {
        "description": "branch0 scoped dynamic-veto token filter only with observed-prior median threshold 1.00 over both overlaps; no one-hop transient subtract",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
            "--ttt_write_replay_token_filter_branch_mask",
            "0",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "linear",
        ],
    },
    "LW34_TTT_OVERLAP_SEMANTIC_HARM_FILTERONLY_B0_PRIOR100": {
        "description": "branch0 semantic-harm scoped token filter only: veto low-prior overlap tokens only when R_ttt marks negative/harm",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_semantic_harm_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
            "--ttt_write_replay_token_filter_branch_mask",
            "0",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "linear",
        ],
    },
    "LW35_TTT_OVERLAP_SEMANTIC_HARM_FILTERONLY_RANDOM_ROLE_B0_PRIOR100": {
        "description": "random same-mass role control for LW34 semantic-harm scoped token filter",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79035",
            "--ttt_write_replay_token_filter_mode",
            "scoped_semantic_harm_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
            "--ttt_write_replay_token_filter_branch_mask",
            "0",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "linear",
        ],
    },
    "LW36_TTT_READHARM_LOCAL_VETO_B0": {
        "description": "READ-harm no-persistent local veto of TTT write prior; allows unrelated stable-positive READ tokens",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_ttt_read_harm_veto_mode",
            "read_harm_no_stable",
            "--semantic_role_ttt_read_harm_veto_scale",
            "0.0",
            "--semantic_role_ttt_read_harm_veto_min_harm",
            "1",
            "--semantic_role_ttt_read_harm_veto_max_stable_positive",
            "999999",
        ],
    },
    "LW37_TTT_READHARM_LOCAL_VETO_RANDOM_ROLE_B0": {
        "description": "random same-mass role control for LW36 READ-harm local no-persistent veto",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79037",
            "--semantic_role_ttt_read_harm_veto_mode",
            "read_harm_no_stable",
            "--semantic_role_ttt_read_harm_veto_scale",
            "0.0",
            "--semantic_role_ttt_read_harm_veto_min_harm",
            "1",
            "--semantic_role_ttt_read_harm_veto_max_stable_positive",
            "999999",
        ],
    },
    "LW38_TTT_READHARM_NEGATIVE_LOCAL_VETO_B0": {
        "description": "narrow READ-harm no-persistent veto of negative TTT role only; keeps neutral/protect write prior",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_ttt_read_harm_veto_mode",
            "read_harm_negative_no_stable",
            "--semantic_role_ttt_read_harm_veto_scale",
            "0.0",
            "--semantic_role_ttt_read_harm_veto_min_harm",
            "1",
            "--semantic_role_ttt_read_harm_veto_max_stable_positive",
            "999999",
        ],
    },
    "LW39_TTT_READHARM_NEGATIVE_LOCAL_VETO_RANDOM_ROLE_B0": {
        "description": "random same-mass role control for LW38 negative-only READ-harm local veto",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79039",
            "--semantic_role_ttt_read_harm_veto_mode",
            "read_harm_negative_no_stable",
            "--semantic_role_ttt_read_harm_veto_scale",
            "0.0",
            "--semantic_role_ttt_read_harm_veto_min_harm",
            "1",
            "--semantic_role_ttt_read_harm_veto_max_stable_positive",
            "999999",
        ],
    },
    "LW40_TTT_V80_HEAD_SUPPORT_VETO_B0": {
        "description": "v80 geometry-error semantic low-support head-overlap TTT replay veto on branch0",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_ttt_overlap_support_dir",
            "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final/phase9_seq01_ref055_v80_error_semantic_support_maps",
            "--semantic_ttt_overlap_support_score_key",
            "score_overlap",
            "--semantic_ttt_overlap_support_scope",
            "head_overlap",
            "--semantic_ttt_overlap_support_floor",
            "0.0",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "head_overlap",
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
    "LW41_TTT_V80_HEAD_SUPPORT_RANDOM_VETO_B0": {
        "description": "random support-map control for LW40 v80 head-overlap support replay veto",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "diagnostic_control": True,
        "extra": [
            "--semantic_ttt_overlap_support_dir",
            "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final/phase9_seq01_ref055_v80_error_semantic_support_maps",
            "--semantic_ttt_overlap_support_score_key",
            "control_overlap",
            "--semantic_ttt_overlap_support_scope",
            "head_overlap",
            "--semantic_ttt_overlap_support_floor",
            "0.0",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "head_overlap",
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
    "LW42_TTT_V80_SELECTED_WRITE_SUPPORT_VETO_B0": {
        "description": "v80 selected-write low-support head-overlap TTT replay veto on branch0",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_ttt_overlap_support_dir",
            "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final/phase9_seq01_ref055_v80_selected_write_support_maps",
            "--semantic_ttt_overlap_support_score_key",
            "score_overlap",
            "--semantic_ttt_overlap_support_scope",
            "head_overlap",
            "--semantic_ttt_overlap_support_floor",
            "0.0",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "head_overlap",
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
    "LW43_TTT_V80_SELECTED_WRITE_SUPPORT_RANDOM_VETO_B0": {
        "description": "same-mass low-support random control for LW42 selected-write support replay veto",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "diagnostic_control": True,
        "extra": [
            "--semantic_ttt_overlap_support_dir",
            "results/kitti01_hmc_v2/acl2_v80tf_multiseq_goodbad_semantic_three_memory_control/report_final/phase9_seq01_ref055_v80_selected_write_support_maps",
            "--semantic_ttt_overlap_support_score_key",
            "control_overlap",
            "--semantic_ttt_overlap_support_scope",
            "head_overlap",
            "--semantic_ttt_overlap_support_floor",
            "0.0",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "head_overlap",
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
    "LW21_TTT_OVERLAP_DYNAMIC_TTL_B0_PRIOR108": {
        "description": "branch0 one-hop dynamic TTL with observed-prior top-decile threshold 1.08 over both overlaps",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
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
    "LW22_TTT_OVERLAP_DYNAMIC_TTL_RANDOM_ROLE_B0_PRIOR108": {
        "description": "random same-mass role control for LW21 observed-prior top-decile threshold TTL",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79022",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
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
    "LW23_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR100": {
        "description": "semantic role-scaled branch0 one-hop TTL with prior threshold 1.00; stable mild boost, context neutral, harmful damp",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
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
    "LW24_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR100": {
        "description": "random same-mass role control for LW23 role-scaled TTL threshold 1.00",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79024",
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.00",
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
    "LW25_TTT_OVERLAP_ROLECAL_TTL_B0_PRIOR108": {
        "description": "semantic role-scaled branch0 one-hop TTL with prior threshold 1.08; stricter top-decile filter",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
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
    "LW26_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B0_PRIOR108": {
        "description": "random same-mass role control for LW25 role-scaled TTL threshold 1.08",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79026",
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
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
    "LW27_TTT_OVERLAP_ROLESTRONG_TTL_B0_PRIOR108": {
        "description": "stronger harmful-damp semantic role-scaled branch0 one-hop TTL with prior threshold 1.08",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.65",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
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
    "LW28_TTT_OVERLAP_ROLESTRONG_TTL_RANDOM_ROLE_B0_PRIOR108": {
        "description": "random same-mass role control for LW27 stronger harmful-damp TTL threshold 1.08",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79028",
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.65",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
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
    "LW29_TTT_OVERLAP_ROLECAL_TTL_B1_PRIOR108": {
        "description": "semantic role-scaled branch1 one-hop TTL with prior threshold 1.08; branch-layer attribution after branch0 tail-future conflict",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
            "--ttt_write_replay_token_filter_branch_mask",
            "1",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "ttl_dynamic",
            "--ttt_write_transient_delta_subtract_scale",
            "1.0",
            "--ttt_write_transient_delta_branch_mask",
            "1",
            "--ttt_write_transient_delta_ttl",
            "1",
        ],
    },
    "LW30_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B1_PRIOR108": {
        "description": "random same-mass role control for LW29 branch1 role-scaled TTL threshold 1.08",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79030",
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
            "--ttt_write_replay_token_filter_branch_mask",
            "1",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "ttl_dynamic",
            "--ttt_write_transient_delta_subtract_scale",
            "1.0",
            "--ttt_write_transient_delta_branch_mask",
            "1",
            "--ttt_write_transient_delta_ttl",
            "1",
        ],
    },
    "LW31_TTT_OVERLAP_ROLECAL_TTL_B2_PRIOR108": {
        "description": "semantic role-scaled branch2 one-hop TTL with prior threshold 1.08; branch-layer attribution after branch0 tail-future conflict",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "extra": [
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
            "--ttt_write_replay_token_filter_branch_mask",
            "2",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "ttl_dynamic",
            "--ttt_write_transient_delta_subtract_scale",
            "1.0",
            "--ttt_write_transient_delta_branch_mask",
            "2",
            "--ttt_write_transient_delta_ttl",
            "1",
        ],
    },
    "LW32_TTT_OVERLAP_ROLECAL_TTL_RANDOM_ROLE_B2_PRIOR108": {
        "description": "random same-mass role control for LW31 branch2 role-scaled TTL threshold 1.08",
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "random_same_mass",
        "diagnostic_control": True,
        "extra": [
            "--semantic_role_control_seed",
            "79032",
            "--semantic_role_positive_scale",
            "1.05",
            "--semantic_role_neutral_scale",
            "1.0",
            "--semantic_role_negative_scale",
            "0.85",
            "--ttt_write_replay_token_filter_mode",
            "scoped_dynamic_veto",
            "--ttt_write_replay_token_filter_scope",
            "both_overlap",
            "--ttt_write_replay_token_filter_threshold",
            "1.08",
            "--ttt_write_replay_token_filter_branch_mask",
            "2",
            "--ttt_write_replay_token_filter_blend",
            "1.0",
            "--ttt_write_replay_token_filter_blend_mode",
            "ttl_dynamic",
            "--ttt_write_transient_delta_subtract_scale",
            "1.0",
            "--ttt_write_transient_delta_branch_mask",
            "2",
            "--ttt_write_transient_delta_ttl",
            "1",
        ],
    },
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return value


def _parse_ints(text: str) -> list[int]:
    text = str(text or "").strip()
    if not text:
        return []
    if "-" in text and "," not in text:
        left, right = text.split("-", 1)
        a, b = int(left), int(right)
        if b < a:
            raise ValueError(f"bad range: {text!r}")
        return list(range(a, b + 1))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _safe_improvement(base: Any, cand: Any) -> float | None:
    b = _finite(base)
    c = _finite(cand)
    if b is None or c is None or abs(b) < 1e-12:
        return None
    return float((b - c) / abs(b))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), ensure_ascii=False)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[Any]) -> float | None:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.mean(xs)) if xs else None


def _sum(values: list[Any]) -> float | None:
    xs = [_finite(v) for v in values]
    xs = [v for v in xs if v is not None]
    return float(np.sum(xs)) if xs else None


def _aggregate_hmc(run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl(run_dir / "hmc_state_hash.jsonl")
    out: dict[str, Any] = {"hmc_rows": int(len(rows))}
    if not rows:
        return out
    prefixes = (
        "ttt_write_commit_filter_",
        "ttt_write_commit_ema_",
        "ttt_write_delta_",
        "ttt_write_native_mix_",
        "ttt_write_prior_transform_",
        "ttt_write_transient_delta_",
        "ttt_transient_delta_",
        "ttt_replay_token_filter_",
        "ttt_replay_feature_gate_",
        "ttt_replay_feature_",
        "ttt_write_native_delta_gate_",
        "ttt_gradient_reversal_",
    )
    keys = sorted({k for row in rows for k in row if str(k).startswith(prefixes)})
    out["ttt_debug_keys_present"] = keys
    for key in keys:
        vals = [row.get(key) for row in rows if row.get(key) is not None]
        if not vals:
            continue
        if all(isinstance(v, bool) for v in vals):
            out[f"{key}_true_count"] = int(sum(1 for v in vals if bool(v)))
            out[f"{key}_true_frac"] = float(sum(1 for v in vals if bool(v)) / len(vals))
        elif all(_finite(v) is not None for v in vals):
            out[f"{key}_mean"] = _mean(vals)
            out[f"{key}_sum"] = _sum(vals)
        else:
            uniq = []
            seen = set()
            for value in vals:
                token = json.dumps(_jsonable(value), sort_keys=True, ensure_ascii=False)
                if token not in seen:
                    seen.add(token)
                    uniq.append(value)
            out[f"{key}_set"] = uniq[:16]
    out["memory_ttt_mean_rel_diff_mean"] = _mean([row.get("memory_ttt_mean_rel_diff") for row in rows])
    out["memory_ttt_max_rel_diff_mean"] = _mean([row.get("memory_ttt_max_rel_diff") for row in rows])
    out["state_double_write_safe_all"] = bool(all(bool(row.get("state_double_write_safe", True)) for row in rows))
    return out


def _window_bounds(chunks: list[int], chunk_size: int, overlap: int) -> dict[str, int]:
    if not chunks:
        raise ValueError("--chunks must not be empty")
    stride = int(chunk_size) - int(overlap)
    if stride <= 0:
        raise ValueError("chunk_size must be larger than overlap")
    expected = list(range(min(chunks), max(chunks) + 1))
    if chunks != expected:
        raise ValueError(f"--chunks must be contiguous and sorted; got {chunks}")
    start_frame = chunks[0] * stride
    end_frame = start_frame + int(chunk_size) + (len(chunks) - 1) * stride
    return {
        "first_chunk": int(chunks[0]),
        "last_chunk": int(chunks[-1]),
        "num_chunks": int(len(chunks)),
        "stride": int(stride),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
    }


def _build_command(args: argparse.Namespace, case_name: str, out_dir: Path, bounds: dict[str, int]) -> list[str]:
    case = CASES[case_name]
    seq = str(args.seq).zfill(2)
    cmd = [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        str(args.conda_env),
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.data_root / "sequences" / seq / "image_2"),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{seq}.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        str(bounds["start_frame"]),
        "--end_frame",
        str(bounds["end_frame"]),
        "--global_chunk_offset",
        str(bounds["first_chunk"]),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        str(case["hybrid_memory_mode"]),
        "--hmc_commit_mode",
        str(case["hmc_commit_mode"]),
        "--semantic_prior_mode",
        str(case["semantic_prior_mode"]),
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--enable_frame_read_control",
        "0",
        "--read_path",
        "none",
        "--read_cue_source",
        "dyn",
        "--semantic_role_policy",
        str(case["semantic_role_policy"]),
        "--semantic_memory_paths",
        str(case["semantic_memory_paths"]),
        "--semantic_role_control_mode",
        str(case["semantic_role_control_mode"]),
        "--semantic_role_positive_scale",
        "1.0",
        "--semantic_role_neutral_scale",
        "1.0",
        "--semantic_role_negative_scale",
        "1.0",
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    cmd.extend(str(x) for x in case.get("extra", []))
    if bool(case.get("freeze_all")):
        local_chunks = ",".join(str(i) for i in range(int(bounds["num_chunks"])))
        cmd.extend(["--ttt_freeze_chunks", local_chunks])
    return cmd


def _run_job(job: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
    if bool(job.get("disable_ttt_compile", False)):
        env["LOGER_TTT_DISABLE_COMPILE"] = "1"
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    start_t = time.time()
    with run_log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            job["cmd"],
            cwd=job["workdir"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    job.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(time.time() - start_t),
            "run_log": str(run_log),
            "trajectory": str(out_dir / f"{str(job['seq']).zfill(2)}.txt"),
            "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    return job


def _evaluate(args: argparse.Namespace, jobs: list[dict[str, Any]], bounds: dict[str, int]) -> dict[str, Any]:
    seq = str(args.seq).zfill(2)
    metric_rows: list[dict[str, Any]] = []
    for job in jobs:
        case_name = str(job["case"])
        out_dir = Path(job["out_dir"])
        row: dict[str, Any] = {
            "case": case_name,
            "description": CASES[case_name].get("description"),
            "diagnostic_control": bool(CASES[case_name].get("diagnostic_control", False)),
            "returncode": int(job.get("returncode") or 0),
            "trajectory": str(out_dir / f"{seq}.txt"),
            "run_log": str(out_dir / "run.log"),
        }
        if int(job.get("returncode") or 0) == 0:
            _single, _pairs, window5, _summary = _evaluate_run(
                name=case_name,
                seq=seq,
                path=out_dir / f"{seq}.txt",
                gt_root=args.data_root / "poses",
                chunk_size=int(args.chunk_size),
                overlap=int(args.chunk_overlap),
                min_coverage=float(args.min_coverage),
            )
            target = None
            target_chunks = "-".join(str(c) for c in _parse_ints(args.chunks))
            for item in window5:
                if str(item.get("window_chunks")) == target_chunks:
                    target = item
                    break
            if target is None and window5:
                target = window5[0]
            if target is not None:
                row.update(target)
            row.update(_aggregate_hmc(out_dir))
        metric_rows.append(row)

    by_case = {str(row["case"]): row for row in metric_rows}
    baseline = str(args.baseline)
    native = str(args.native_baseline)
    base_rmse = by_case.get(baseline, {}).get("window5_joint_sim3_rmse_m")
    native_rmse = by_case.get(native, {}).get("window5_joint_sim3_rmse_m")
    decisions: dict[str, Any] = {}
    for row in metric_rows:
        case_name = str(row["case"])
        if case_name in {baseline, native}:
            continue
        cand_rmse = row.get("window5_joint_sim3_rmse_m")
        improve_vs_ttt = _safe_improvement(base_rmse, cand_rmse)
        improve_vs_native = _safe_improvement(native_rmse, cand_rmse)
        decisions[case_name] = {
            "candidate": case_name,
            "window5_joint_sim3_rmse_m": cand_rmse,
            "baseline": baseline,
            "baseline_window5_joint_sim3_rmse_m": base_rmse,
            "native_baseline": native,
            "native_window5_joint_sim3_rmse_m": native_rmse,
            "improvement_vs_ttt_baseline_ratio": improve_vs_ttt,
            "improvement_vs_native_ratio": improve_vs_native,
            "single_window_smoke_improves_vs_ttt_baseline": bool(
                improve_vs_ttt is not None and improve_vs_ttt > 0.0
            ),
            "single_window_smoke_improves_ge_min_ratio": bool(
                improve_vs_ttt is not None and improve_vs_ttt >= float(args.min_improvement)
            ),
            "diagnostic_control": bool(row.get("diagnostic_control", False)),
            "method_gate_claimed": False,
        }

    summary = {
        "schema": "acl2_v78_ttt_long_window_regime_action_smoke_v1",
        "diagnostic_only": True,
        "method_gate_claimed": False,
        "runtime_promotion_allowed": False,
        "runtime_promotion_blocker": (
            "single-window smoke only; requires held-out five-chunk windows and "
            "window5_joint_sim3/downstream consistency improvement before promotion"
        ),
        "seq": seq,
        "chunks": _parse_ints(args.chunks),
        "window_bounds": bounds,
        "cases": {name: CASES[name] for name in by_case},
        "baseline": baseline,
        "native_baseline": native,
        "min_improvement": float(args.min_improvement),
        "metrics": metric_rows,
        "decisions": decisions,
        "any_single_window_improves_vs_ttt_baseline": bool(
            any(bool(d.get("single_window_smoke_improves_vs_ttt_baseline")) for d in decisions.values())
        ),
        "any_single_window_improves_ge_min_ratio": bool(
            any(bool(d.get("single_window_smoke_improves_ge_min_ratio")) for d in decisions.values())
        ),
    }
    metrics_json = args.output_root / "long_window_ttt_regime_action_metrics.json"
    metrics_csv = args.output_root / "long_window_ttt_regime_action_metrics.csv"
    decision_json = args.output_root / "long_window_ttt_regime_action_decision.json"
    metrics_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_json.write_text(
        json.dumps(
            _jsonable(
                {
                    "diagnostic_only": True,
                    "method_gate_claimed": False,
                    "runtime_promotion_allowed": False,
                    "decisions": decisions,
                    "any_single_window_improves_vs_ttt_baseline": summary[
                        "any_single_window_improves_vs_ttt_baseline"
                    ],
                    "any_single_window_improves_ge_min_ratio": summary[
                        "any_single_window_improves_ge_min_ratio"
                    ],
                }
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_csv, metric_rows)
    print(
        json.dumps(
            _jsonable(
                {
                    "any_single_window_improves_vs_ttt_baseline": summary[
                        "any_single_window_improves_vs_ttt_baseline"
                    ],
                    "any_single_window_improves_ge_min_ratio": summary[
                        "any_single_window_improves_ge_min_ratio"
                    ],
                    "runtime_promotion_allowed": False,
                    "metrics_json": metrics_json,
                    "decision_json": decision_json,
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
    parser.add_argument("--seq", default="02")
    parser.add_argument("--chunks", default="64-68", help="Contiguous chunk ids, e.g. 64-68 or 64,65,66,67,68.")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=None)
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
    parser.add_argument("--min-improvement", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seq = str(args.seq).zfill(2)
    chunks = _parse_ints(args.chunks)
    bounds = _window_bounds(chunks, int(args.chunk_size), int(args.chunk_overlap))
    gpus = _parse_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if args.stage_c_cache_dir is None:
        args.stage_c_cache_dir = Path(f"results/kitti_preprocess/{seq}/stage_c_cache_semantic_chunks")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(CASES)}")
    for required in (args.baseline, args.native_baseline):
        if required not in cases:
            raise ValueError(f"required baseline {required!r} is not included in --cases")

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for idx, case_name in enumerate(cases):
        out_dir = args.output_root / case_name
        cmd = _build_command(args, case_name, out_dir, bounds)
        traj = out_dir / f"{seq}.txt"
        hmc = out_dir / "hmc_state_hash.jsonl"
        skipped = bool(args.skip_existing and traj.exists() and hmc.exists())
        jobs.append(
            {
                "seq": seq,
                "case": case_name,
                "case_config": CASES[case_name],
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
        "schema": "acl2_v78_ttt_long_window_regime_action_smoke_manifest_v1",
        "diagnostic_only": True,
        "args": _jsonable(vars(args)),
        "window_bounds": bounds,
        "jobs": jobs,
    }
    manifest_path = args.output_root / "long_window_ttt_regime_action_run_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: list[dict[str, Any]] = [job for job in jobs if job["skipped"]]
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
                f"finished case={result['case']} gpu={result['gpu']} "
                f"returncode={result['returncode']} duration_sec={result['duration_sec']:.1f}"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(gpus))) as pool:
        futures = [pool.submit(run_gpu_queue, queue) for queue in jobs_by_gpu.values() if queue]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    completed_by_case = {str(job["case"]): job for job in completed}
    ordered_completed = [completed_by_case.get(str(job["case"]), job) for job in jobs]
    run_summary = {
        "schema": "acl2_v78_ttt_long_window_regime_action_smoke_run_summary_v1",
        "diagnostic_only": True,
        "jobs": ordered_completed,
        "all_jobs_ok": bool(all(int(job.get("returncode") or 0) == 0 for job in ordered_completed)),
    }
    run_summary_path = args.output_root / "long_window_ttt_regime_action_run_summary.json"
    run_summary_path.write_text(json.dumps(_jsonable(run_summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"all_jobs_ok={run_summary['all_jobs_ok']} run_summary={run_summary_path}")

    if not args.no_evaluate:
        _evaluate(args, ordered_completed, bounds)


if __name__ == "__main__":
    main()
