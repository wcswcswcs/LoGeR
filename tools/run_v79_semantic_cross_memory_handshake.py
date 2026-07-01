#!/usr/bin/env python3
"""Run ACL2 v79 Phase5 cross-memory semantic handshake smokes.

The runner keeps the Phase5 question narrow and auditable: can the Phase2
READ1 semantic signal propagate into a TTT write/update action on the hardest
five-chunk target, and beat READ-only, TTT-only, geometry-only, and random-role
controls?  Direct READ->TTT role intersection is required for a strict Phase5
gate; when that hook is not present the run is still useful, but the gate stays
false instead of inventing an alignment score.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import (  # noqa: E402
    LOWER_IS_BETTER_KEYS,
    _eval_run,
    _finite,
    _load_kitti_gt,
    _safe_ratio_improvement,
    _write_csv,
)
from tools.run_v78_phase4_ttt_write_role_control import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONDA,
    DEFAULT_CONFIG,
    DEFAULT_CUDA_ALLOC_CONF,
    DEFAULT_GT,
    DEFAULT_INPUT,
    DEFAULT_STAGE_C_CACHE,
    _aggregate_phase4_hmc,
    _context_window,
    _jsonable,
    _parse_csv_ints,
)


DEFAULT_OUTPUT_ROOT = Path(
    "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/"
    "report_final/phase5_cross_memory_semantic_handshake/read_to_ttt_fivechunk_7_11"
)

MECHANISM_KEYS = [
    "head10_to_tail10_pose_sim3_rmse_m",
    "overlap3_to_future_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]

PHASE5_CASES: Dict[str, Dict[str, Any]] = {
    "HS0_NATIVE_TTT_PROBE": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "noop",
        "read_path": "none",
        "read_cue_source": "dyn",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.0,
        "neutral_scale": 1.0,
        "negative_scale": 1.0,
        "semantic_contract": "native TTT probe baseline; no semantic read or write role",
    },
    "HS1_READ_ONLY_BEST_SEM": {
        "hybrid_memory_mode": "read_path_only",
        "hmc_commit_mode": "controlled",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.0,
        "neutral_scale": 1.0,
        "negative_scale": 1.0,
        "semantic_contract": "Phase2 winning READ1 semantic layout read only",
    },
    "HS3_TTT_ONLY_SEM": {
        "hybrid_memory_mode": "ttt_write_only",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "none",
        "read_cue_source": "dyn",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": "TTT-only semantic write role from Phase4 T3",
    },
    "HS5_READ_TO_TTT_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": "READ1 semantic layout signal plus TTT semantic write role",
    },
    "HS6_READ_ACTIVE_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_structure_positive_highd_short",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": "READ active structure becomes positive TTT carrier; READ active high-D nonstructure remains short negative",
    },
    "HS7_READ_ACTIVE_TTT_POS_NEUTRAL_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_structure_positive_context_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": "READ active structure becomes positive TTT carrier; other READ-active high-trust context is neutral instead of short-negative",
    },
    "HS10_READ_ACTIVE_ALL_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_all_positive",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": "All READ-active high-trust evidence becomes positive TTT carrier to test whether weak READ signal needs larger positive mass",
    },
    "HS11_READ_ACTIVE_REGIME_GUARDED_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_regime_guarded_pos_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": (
            "READ-active low-D high-trust structure becomes positive TTT carrier; "
            "other READ-active high-trust evidence is neutral to guard long-window regime shifts"
        ),
    },
    "HS12_READ_ACTIVE_KEY_STABLE_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_key_stable_positive_context_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "v68_frame_attn_debug": True,
        "semantic_contract": (
            "READ-active high-trust evidence becomes TTT positive only when it also has high K-side "
            "stability key_avg*(1-qk_var); other READ-active high-trust evidence is neutral"
        ),
    },
    "HS13_READ_ACTIVE_KEY_STABLE_LOWD_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_key_stable_lowd_positive_context_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "v68_frame_attn_debug": True,
        "semantic_contract": (
            "READ-active high-trust evidence becomes TTT positive only when it has both high K-side "
            "stability key_avg*(1-qk_var) and low-D regime evidence; other READ-active high-trust evidence is neutral"
        ),
    },
    "HS14_READ_ACTIVE_POS_NEUTRAL_TAIL_DECAY_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_structure_positive_context_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--ttt_write_commit_filter_mode",
            "old_decay_by_risk",
            "--ttt_write_commit_filter_risk_source",
            "d_tok",
            "--ttt_write_commit_filter_scope",
            "tail_overlap",
            "--ttt_write_commit_filter_stat",
            "q75",
            "--ttt_write_commit_filter_base",
            "1.0",
            "--ttt_write_commit_filter_gain",
            "0.8",
            "--ttt_write_commit_filter_min",
            "0.35",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "0",
        ],
        "semantic_contract": (
            "HS7 READ-active positive/neutral role handoff, but high-D tail-overlap TTT branch0 "
            "commit is decayed so regime-shift evidence can affect the current chunk without being persistently handed forward"
        ),
    },
    "HS15_READ_ACTIVE_POS_NEUTRAL_TAIL_RESXDG_DECAY_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_structure_positive_context_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--ttt_write_commit_filter_mode",
            "old_decay_by_risk",
            "--ttt_write_commit_filter_risk_source",
            "ttt_residual_x_dg",
            "--ttt_write_commit_filter_scope",
            "tail_overlap",
            "--ttt_write_commit_filter_stat",
            "q75",
            "--ttt_write_commit_filter_base",
            "1.0",
            "--ttt_write_commit_filter_gain",
            "0.8",
            "--ttt_write_commit_filter_min",
            "0.35",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "0",
        ],
        "semantic_contract": (
            "HS7 READ-active positive/neutral role handoff with persistent branch0 commit decay driven by "
            "TTT residual times dynamic/regime risk, repairing the HS14 d_tok tail-overlap no-op risk source"
        ),
    },
    "HS16_READ_ACTIVE_POS_NEUTRAL_TAIL_UPDATECONFLICT_DECAY_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_read_active_structure_positive_context_neutral",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--ttt_write_commit_filter_mode",
            "old_decay_by_risk",
            "--ttt_write_commit_filter_risk_source",
            "update_conflict_energy",
            "--ttt_write_commit_filter_scope",
            "tail_overlap",
            "--ttt_write_commit_filter_stat",
            "q75",
            "--ttt_write_commit_filter_base",
            "1.0",
            "--ttt_write_commit_filter_gain",
            "0.8",
            "--ttt_write_commit_filter_min",
            "0.35",
            "--ttt_write_commit_filter_max",
            "1.0",
            "--ttt_write_commit_filter_branch_mask",
            "0",
        ],
        "semantic_contract": (
            "HS7 READ-active positive/neutral role handoff with persistent branch0 commit decay driven by "
            "TTT update-conflict energy, testing whether harmful long-window persistence is exposed by the write update itself"
        ),
    },
    "HS17_SWA_STABLE_TOP25_REDIRECT_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_positive_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_action_active_chunks",
            "9",
        ],
        "semantic_contract": (
            "Phase3 SWA stable-positive top25 overlap support redirects the chunk9 stable carrier: "
            "artifact-selected support becomes TTT positive plus frame/global positive and SWA protect; "
            "other READ-active high-trust context is neutral"
        ),
    },
    "HS18_RANDOM_SWA_STABLE_TOP25_REDIRECT_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_positive_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_swa_redirection_random_control",
            "1",
            "--semantic_action_active_chunks",
            "9",
        ],
        "semantic_contract": "Random same-mass Phase3 SWA stable top25 artifact control for HS17 redirection",
    },
    "HS19_SWA_STABLE_TOP25_SOURCEBOOST_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_action_active_chunks",
            "9",
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "bias_boost",
            "--context_source_skip_scope",
            "both",
            "--context_source_skip_mode",
            "boost",
            "--context_source_skip_mask",
            "swa_redirection_source_positive",
            "--context_source_skip_layer_mode",
            "all",
            "--context_source_skip_soft_rho",
            "0.35",
            "--context_source_skip_record_attention_mass",
            "1",
        ],
        "semantic_contract": (
            "Phase3 SWA stable-positive top25 support is the only frame/global positive source mask; "
            "frame and chunk attention boost artifact-positive source columns while TTT persists only the same artifact carrier"
        ),
    },
    "HS20_RANDOM_SWA_STABLE_TOP25_SOURCEBOOST_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_swa_redirection_random_control",
            "1",
            "--semantic_action_active_chunks",
            "9",
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "bias_boost",
            "--context_source_skip_scope",
            "both",
            "--context_source_skip_mode",
            "boost",
            "--context_source_skip_mask",
            "swa_redirection_source_positive",
            "--context_source_skip_layer_mode",
            "all",
            "--context_source_skip_soft_rho",
            "0.35",
            "--context_source_skip_record_attention_mass",
            "1",
        ],
        "semantic_contract": "Random same-mass artifact source-boost and TTT-write control for HS19",
    },
    "HS21_SWA_STABLE_TOP25_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_action_active_chunks",
            "9",
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "bias_boost",
            "--context_source_skip_scope",
            "both",
            "--context_source_skip_mode",
            "boost",
            "--context_source_skip_mask",
            "swa_redirection_source_positive",
            "--context_source_skip_query_region",
            "mid_tail",
            "--context_source_skip_layer_mode",
            "all",
            "--context_source_skip_soft_rho",
            "0.35",
            "--context_source_skip_record_attention_mass",
            "1",
        ],
        "semantic_contract": (
            "Query-conditioned version of HS19: SWA stable-positive top25 K/V sources are boosted only for "
            "mid/tail query frames so overlap/head structure anchors can be retrieved by downstream chunk queries"
        ),
    },
    "HS22_RANDOM_SWA_STABLE_TOP25_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_swa_redirection_random_control",
            "1",
            "--semantic_action_active_chunks",
            "9",
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "bias_boost",
            "--context_source_skip_scope",
            "both",
            "--context_source_skip_mode",
            "boost",
            "--context_source_skip_mask",
            "swa_redirection_source_positive",
            "--context_source_skip_query_region",
            "mid_tail",
            "--context_source_skip_layer_mode",
            "all",
            "--context_source_skip_soft_rho",
            "0.35",
            "--context_source_skip_record_attention_mass",
            "1",
        ],
        "semantic_contract": "Random same-mass query-conditioned artifact source-boost control for HS21",
    },
    "HS23_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_action_active_chunks",
            "9",
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "bias_boost",
            "--context_source_skip_scope",
            "both",
            "--context_source_skip_mode",
            "boost",
            "--context_source_skip_mask",
            "swa_redirection_source_positive",
            "--context_source_skip_query_region",
            "mid_tail",
            "--context_source_skip_source_attention_top_quantile",
            "0.75",
            "--context_source_skip_layer_mode",
            "all",
            "--context_source_skip_soft_rho",
            "0.35",
            "--context_source_skip_record_attention_mass",
            "1",
        ],
        "semantic_contract": (
            "Query-key compatible source boost: inside SWA stable-positive top25 sources, select the top "
            "mid/tail-query attended source subset at runtime, then boost only that subset while TTT persists the same carrier"
        ),
    },
    "HS24_RANDOM_SWA_STABLE_TOP25_QKATTN_SOURCEBOOST_MIDTAILQ_TTT_POS_SEM": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_swa_stable_top25_source_boost_context_neutral",
        "semantic_memory_paths": "frame,global,swa,ttt",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "extra_cli": [
            "--semantic_swa_redirection_phase3_root",
            "results/kitti01_hmc_v2/acl2_v79tf_revised_semantic_three_memory_control/report_final/phase3_semantic_swa_handoff/source_side_phase9_subset_chunk09",
            "--semantic_swa_redirection_stable_case",
            "P9_40_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_V_LAST",
            "--semantic_swa_redirection_random_stable_case",
            "P9_41_SOURCE_REPLACE_STABLE_AGREEMENT_TOPQ80_RANDOM_SAME_MASS_V_LAST",
            "--semantic_swa_redirection_layer",
            "3",
            "--semantic_swa_redirection_top_quantile",
            "0.75",
            "--semantic_swa_redirection_overlap_frames",
            "3",
            "--semantic_swa_redirection_random_control",
            "1",
            "--semantic_action_active_chunks",
            "9",
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            "bias_boost",
            "--context_source_skip_scope",
            "both",
            "--context_source_skip_mode",
            "boost",
            "--context_source_skip_mask",
            "swa_redirection_source_positive",
            "--context_source_skip_query_region",
            "mid_tail",
            "--context_source_skip_source_attention_top_quantile",
            "0.75",
            "--context_source_skip_layer_mode",
            "all",
            "--context_source_skip_soft_rho",
            "0.35",
            "--context_source_skip_record_attention_mass",
            "1",
        ],
        "semantic_contract": "Random same-mass source mask control for HS23 query-key compatible source boost",
    },
    "HS8_GEOMETRY_ONLY_HANDSHAKE": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v68.read.global_v.l13.geometry_only",
        "semantic_role_policy": "none",
        "semantic_memory_paths": "",
        "semantic_role_control_mode": "none",
        "positive_scale": 1.0,
        "neutral_scale": 1.0,
        "negative_scale": 1.0,
        "semantic_contract": "geometry-only read/action control; no semantic success credit",
    },
    "HS9_RANDOM_ROLE_HANDSHAKE": {
        "hybrid_memory_mode": "hybrid",
        "hmc_commit_mode": "probe_ttt_write",
        "semantic_prior_mode": "spg_v2",
        "read_path": "frame",
        "read_cue_source": "v78.l07_l13.l07_action_only",
        "semantic_role_policy": "fine_ttt_lowstuff_highd_short",
        "semantic_memory_paths": "frame,global,ttt",
        "semantic_role_control_mode": "random_same_mass",
        "positive_scale": 1.25,
        "neutral_scale": 1.0,
        "negative_scale": 0.70,
        "semantic_contract": "same READ mass with random same-mass TTT role control",
    },
}


def _read_jsonl_all(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Iterable[Any]) -> Optional[float]:
    xs = [_finite(v) for v in values]
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _build_command(args: argparse.Namespace, *, chunk: int, case: str, out_dir: Path) -> List[str]:
    cfg = PHASE5_CASES[case]
    window = _context_window(args, int(chunk))
    read_path = str(cfg["read_path"])
    cmd = [
        str(args.conda),
        "run",
        "--no-capture-output",
        "-n",
        args.conda_env,
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(args.input),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / "01.txt"),
        "--checkpoint",
        str(args.checkpoint),
        "--config",
        str(args.config),
        "--chunk_size",
        str(args.chunk_size),
        "--chunk_overlap",
        str(args.chunk_overlap),
        "--start_frame",
        str(window["start_frame"]),
        "--end_frame",
        str(window["end_frame"]),
        "--global_chunk_offset",
        str(window["context_start_chunk"]),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        str(cfg["hybrid_memory_mode"]),
        "--hmc_commit_mode",
        str(cfg["hmc_commit_mode"]),
        "--semantic_prior_mode",
        str(cfg["semantic_prior_mode"]),
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(args.stage_c_cache_dir),
        "--stage_c_cache_require_hit",
        "1",
        "--enable_frame_read_control",
        "1" if read_path == "frame" else "0",
        "--read_path",
        read_path,
        "--read_cue_source",
        str(cfg["read_cue_source"]),
        "--semantic_role_policy",
        str(cfg["semantic_role_policy"]),
        "--semantic_memory_paths",
        str(cfg["semantic_memory_paths"]),
        "--semantic_role_control_mode",
        str(cfg["semantic_role_control_mode"]),
        "--semantic_role_control_seed",
        str(args.semantic_role_control_seed),
        "--semantic_role_highd_quantile",
        str(args.semantic_role_highd_quantile),
        "--semantic_role_low_trust",
        str(args.semantic_role_low_trust),
        "--semantic_role_positive_scale",
        str(cfg["positive_scale"]),
        "--semantic_role_neutral_scale",
        str(cfg["neutral_scale"]),
        "--semantic_role_negative_scale",
        str(cfg["negative_scale"]),
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--ttt_spatial_post_delta_map_dump_dir",
        str(out_dir / "ttt_spatial_post_delta_maps"),
        "--ttt_spatial_post_delta_map_dump_dtype",
        "float16",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
    ]
    if bool(getattr(args, "v68_frame_attn_debug", False)) or bool(cfg.get("v68_frame_attn_debug", False)):
        cmd.extend(["--v68_export_frame_attn_debug", "1"])
    extra_cli = cfg.get("extra_cli") or []
    if extra_cli:
        cmd.extend([str(item) for item in extra_cli])
    extra_pipeline_args = shlex.split(str(getattr(args, "extra_pipeline_args", "") or ""))
    if extra_pipeline_args:
        cmd.extend(extra_pipeline_args)
    if bool(getattr(args, "read_cue_patch_dump", False)) and read_path == "frame":
        cmd.extend(
            [
                "--read_cue_patch_dump_dir",
                str(out_dir / "read_cue_patch_dumps"),
                "--read_cue_patch_dump_dtype",
                str(args.read_cue_patch_dump_dtype),
            ]
        )
    if read_path == "frame":
        cmd.extend(
            [
                "--beta_frame",
                str(args.beta_frame),
                "--frame_bias_mode",
                "key",
                "--read_calib_mode",
                "per_frame_quantile",
                "--read_target_mass",
                str(args.read_target_mass),
                "--read_calib_tau",
                str(args.read_calib_tau),
                "--read_blend_lambda",
                str(args.read_blend_lambda),
                "--read_topk_frac",
                str(args.read_topk_frac),
            ]
        )
    return cmd


def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    if str(job.get("cuda_alloc_conf") or "").strip():
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(job["cuda_alloc_conf"]).strip()
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
            "trajectory": str(out_dir / "01.txt"),
            "hmc_state_hash": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    return job


def _aggregate_phase5_handshake(run_dir: Path) -> Dict[str, Any]:
    rows = _read_jsonl_all(run_dir / "hmc_state_hash.jsonl")
    if not rows:
        return {"phase5_hmc_rows": 0}
    direct_keys = [
        "READ_TTT_ROLE_ALIGNMENT_LOG",
        "read_ttt_role_alignment",
        "prior_read_ttt_role_alignment",
        "read_ttt_alignment_score",
        "prior_read_ttt_alignment_score",
    ]
    direct_values = []
    for row in rows:
        for key in direct_keys:
            if row.get(key) is not None:
                direct_values.append(row.get(key))
    return {
        "phase5_hmc_rows": int(len(rows)),
        "phase5_read_active_count": int(sum(bool(row.get("prior_v78_l07_l13_available")) for row in rows)),
        "phase5_read_output_mean_avg": _mean(row.get("prior_v78_l07_l13_output_mean") for row in rows),
        "phase5_read_output_gt050_mass_avg": _mean(row.get("prior_v78_l07_l13_output_gt050_mass") for row in rows),
        "phase5_read_cue_patch_dump_saved_count": int(sum(row.get("prior_read_cue_patch_dump_status") == "saved" for row in rows)),
        "phase5_read_cue_patch_dump_q90_mass_avg": _mean(row.get("prior_read_cue_patch_dump_q90_mass") for row in rows),
        "phase5_read_cue_patch_dump_paths_sample": [
            row.get("prior_read_cue_patch_dump_path")
            for row in rows
            if row.get("prior_read_cue_patch_dump_path")
        ][:8],
        "phase5_ttt_role_consumed_count": int(sum(bool(row.get("prior_ttt_write_present")) for row in rows)),
        "phase5_semantic_role_consumed_count": int(sum(bool(row.get("prior_semantic_role_consumed_any")) for row in rows)),
        "phase5_direct_read_ttt_alignment_available": bool(direct_values),
        "phase5_direct_read_ttt_alignment_values": direct_values[:8],
        "phase5_direct_read_ttt_alignment_score": None,
    }


def _best_value(rows_by_name: Dict[str, Dict[str, Any]], names: Sequence[str], key: str) -> Optional[float]:
    vals = [_finite(rows_by_name.get(name, {}).get(key)) for name in names]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _build_phase5_decision(
    rows: Sequence[Dict[str, Any]],
    *,
    candidate: str,
    baseline: str,
    single_paths: Sequence[str],
    controls: Sequence[str],
    min_improvement: float,
) -> Dict[str, Any]:
    by_name = {str(row["run"]): row for row in rows}
    cand = by_name.get(candidate)
    base = by_name.get(baseline)
    if cand is None:
        return {"phase5_gate_pass": False, "candidate": candidate, "reason": f"missing_candidate:{candidate}"}
    if base is None:
        return {"phase5_gate_pass": False, "candidate": candidate, "reason": f"missing_baseline:{baseline}"}
    direct_align = bool(cand.get("phase5_direct_read_ttt_alignment_available"))

    comparisons: Dict[str, Any] = {}
    metric_passes: List[str] = []
    for key in LOWER_IS_BETTER_KEYS:
        cand_v = _finite(cand.get(key))
        base_v = _finite(base.get(key))
        best_single = _best_value(by_name, single_paths, key)
        best_control = _best_value(by_name, controls, key)
        ratio = _safe_ratio_improvement(base_v, cand_v)
        beats_single = bool(cand_v is not None and best_single is not None and cand_v < best_single)
        beats_controls = bool(cand_v is not None and best_control is not None and cand_v < best_control)
        mechanism_key = key in MECHANISM_KEYS
        key_pass = bool(mechanism_key and beats_single and beats_controls and ratio is not None and ratio >= min_improvement)
        if key_pass:
            metric_passes.append(key)
        comparisons[key] = {
            "candidate": cand_v,
            "baseline": base_v,
            "best_single_path": best_single,
            "best_control": best_control,
            "candidate_minus_baseline": (cand_v - base_v) if cand_v is not None and base_v is not None else None,
            "candidate_minus_best_single": (cand_v - best_single) if cand_v is not None and best_single is not None else None,
            "candidate_minus_best_control": (cand_v - best_control) if cand_v is not None and best_control is not None else None,
            "improvement_vs_baseline_ratio": ratio,
            "beats_best_single_path": beats_single,
            "beats_controls": beats_controls,
            "mechanism_key": mechanism_key,
            "phase5_metric_key_pass": key_pass,
        }
    gate_pass = bool(direct_align and metric_passes)
    blockers: List[str] = []
    if not direct_align:
        blockers.append("missing_direct_READ_TTT_role_alignment_log")
    if not metric_passes:
        blockers.append("no_mechanism_metric_improves_ge_threshold_while_beating_best_single_and_controls")
    return {
        "phase5_gate_pass": gate_pass,
        "candidate": candidate,
        "baseline": baseline,
        "single_paths": list(single_paths),
        "controls": list(controls),
        "metric_passes": metric_passes,
        "direct_read_ttt_alignment_available": direct_align,
        "direct_alignment_rule": "strict Phase5 requires direct READ active role intersect TTT positive/negative role; proxy marginals are not counted",
        "blockers": blockers,
        "comparisons": comparisons,
    }


def _evaluate(args: argparse.Namespace, ordered_jobs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    _, gt_poses_all, gt_pos_all = _load_kitti_gt(args.gt)
    rows: List[Dict[str, Any]] = []
    for job in ordered_jobs:
        if int(job.get("returncode") or 0) != 0:
            continue
        run_name = str(job["case"])
        run_dir = Path(job["out_dir"])
        row = _eval_run(run_name, run_dir, gt_poses_all, gt_pos_all)
        row.update(_aggregate_phase4_hmc(run_dir))
        row.update(_aggregate_phase5_handshake(run_dir))
        rows.append(row)

    single_paths = [case.strip() for case in str(args.single_paths).split(",") if case.strip()]
    controls = [case.strip() for case in str(args.controls).split(",") if case.strip()]
    candidates = [case.strip() for case in str(args.candidates).split(",") if case.strip()]
    decisions = {
        cand: _build_phase5_decision(
            rows,
            candidate=cand,
            baseline=str(args.baseline),
            single_paths=single_paths,
            controls=controls,
            min_improvement=float(args.min_mechanism_improvement),
        )
        for cand in candidates
    }
    summary = {
        "schema": "acl2_v79_phase5_cross_memory_semantic_handshake_summary_v1",
        "output_root": str(args.output_root),
        "baseline": str(args.baseline),
        "single_paths": single_paths,
        "controls": controls,
        "candidates": candidates,
        "runs": rows,
        "decisions": decisions,
        "phase5_any_gate_pass": bool(any(bool(d.get("phase5_gate_pass")) for d in decisions.values())),
    }
    metrics_json = args.output_root / "phase5_cross_memory_handshake_metrics.json"
    metrics_csv = args.output_root / "phase5_cross_memory_handshake_metrics.csv"
    decision_json = args.output_root / "phase5_cross_memory_handshake_decision.json"
    metrics_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    decision_json.write_text(
        json.dumps(_jsonable({"decisions": decisions, "phase5_any_gate_pass": summary["phase5_any_gate_pass"]}), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(metrics_csv, rows)
    print(json.dumps(_jsonable({"phase5_any_gate_pass": summary["phase5_any_gate_pass"], "decisions": decisions}), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"wrote_json={metrics_json}")
    print(f"wrote_csv={metrics_csv}")
    print(f"wrote_decision={decision_json}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunks", default="11")
    parser.add_argument("--cases", default=",".join(PHASE5_CASES))
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--conda", type=Path, default=DEFAULT_CONDA)
    parser.add_argument("--conda-env", default="loger")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage-c-cache-dir", type=Path, default=DEFAULT_STAGE_C_CACHE)
    parser.add_argument("--cuda-alloc-conf", default=DEFAULT_CUDA_ALLOC_CONF)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--chunk-overlap", type=int, default=3)
    parser.add_argument("--context-chunks", type=int, default=5)
    parser.add_argument("--semantic-role-highd-quantile", type=float, default=0.75)
    parser.add_argument("--semantic-role-low-trust", type=float, default=0.55)
    parser.add_argument("--semantic-role-control-seed", type=int, default=7804)
    parser.add_argument("--beta-frame", type=float, default=0.5)
    parser.add_argument("--read-target-mass", type=float, default=0.1)
    parser.add_argument("--read-calib-tau", type=float, default=0.05)
    parser.add_argument("--read-blend-lambda", type=float, default=0.5)
    parser.add_argument("--read-topk-frac", type=float, default=0.1)
    parser.add_argument("--read-cue-patch-dump", action="store_true")
    parser.add_argument("--read-cue-patch-dump-dtype", default="float16")
    parser.add_argument(
        "--extra-pipeline-args",
        default="",
        help="Additional run_pipeline_abc_v2.py arguments appended to every case command.",
    )
    parser.add_argument(
        "--v68-frame-attn-debug",
        action="store_true",
        help="Forward --v68_export_frame_attn_debug=1 to run_pipeline_abc_v2 so frame/key cosine maps are available.",
    )
    parser.add_argument("--baseline", default="HS0_NATIVE_TTT_PROBE")
    parser.add_argument("--single-paths", default="HS1_READ_ONLY_BEST_SEM,HS3_TTT_ONLY_SEM")
    parser.add_argument("--controls", default="HS8_GEOMETRY_ONLY_HANDSHAKE,HS9_RANDOM_ROLE_HANDSHAKE")
    parser.add_argument("--candidates", default="HS5_READ_TO_TTT_SEM")
    parser.add_argument("--min-mechanism-improvement", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    args = parser.parse_args()

    chunks = _parse_csv_ints(args.chunks)
    gpus = _parse_csv_ints(args.gpus)
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    cases = [case.strip() for case in str(args.cases).split(",") if case.strip()]
    unknown = [case for case in cases if case not in PHASE5_CASES]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}; known={sorted(PHASE5_CASES)}")

    jobs: List[Dict[str, Any]] = []
    gpu_cursor = 0
    for chunk in chunks:
        window = _context_window(args, int(chunk))
        for case in cases:
            out_dir = args.output_root / f"chunk{int(chunk):02d}" / case
            cmd = _build_command(args, chunk=int(chunk), case=case, out_dir=out_dir)
            skipped = bool(args.skip_existing and (out_dir / "01.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            jobs.append(
                {
                    "chunk": int(chunk),
                    **window,
                    "case": case,
                    "case_config": PHASE5_CASES[case],
                    "gpu": int(gpus[gpu_cursor % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "workdir": str(args.workdir),
                    "cuda_alloc_conf": str(args.cuda_alloc_conf),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
            gpu_cursor += 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "phase5_cross_memory_handshake_run_manifest.json"
    manifest: Dict[str, Any] = {"args": _jsonable(vars(args)), "jobs": jobs}
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"planned_jobs={len(jobs)} manifest={manifest_path}")
    if args.dry_run:
        return

    run_jobs = [job for job in jobs if not job["skipped"]]
    completed: List[Dict[str, Any]] = [job for job in jobs if job["skipped"]]
    completed_lock = threading.Lock()
    jobs_by_gpu: Dict[int, List[Dict[str, Any]]] = {int(gpu): [] for gpu in gpus}
    for job in run_jobs:
        jobs_by_gpu[int(job["gpu"])].append(job)

    def _run_gpu_queue(gpu: int, queue: List[Dict[str, Any]]) -> None:
        for job in queue:
            result = _run_job(job)
            with completed_lock:
                completed.append(result)
                manifest["jobs"] = completed + [item for item in jobs if item not in completed]
                manifest_path.write_text(
                    json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[gpu{gpu}] chunk={result['chunk']} case={result['case']} "
                f"returncode={result['returncode']} duration={result['duration_sec']:.1f}s",
                flush=True,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(_run_gpu_queue, int(gpu), queue)
            for gpu, queue in jobs_by_gpu.items()
            if queue
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    completed_by_key = {(int(job["chunk"]), str(job["case"])): job for job in completed}
    ordered = [completed_by_key.get((int(job["chunk"]), str(job["case"])), job) for job in jobs]
    failed = [job for job in ordered if int(job.get("returncode") or 0) != 0]
    manifest["jobs"] = ordered
    manifest["completed_count"] = int(len([job for job in ordered if job.get("returncode") is not None]))
    manifest["failed_jobs"] = [
        {"chunk": int(job["chunk"]), "case": str(job["case"]), "returncode": int(job.get("returncode") or -1)}
        for job in failed
    ]
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"completed={manifest['completed_count']} failed={len(failed)} manifest={manifest_path}")
    if not bool(args.no_evaluate):
        _evaluate(args, ordered)


if __name__ == "__main__":
    main()
