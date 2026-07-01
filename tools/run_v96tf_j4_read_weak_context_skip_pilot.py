#!/usr/bin/env python3
"""Run ACL2 v96 Track J4 READ weak-context early K-side skip pilot.

This is a short-window mechanism pilot, not Stage7 full validation.  It uses
the v96 J3 READ candidate (WEAK_SCALE_CONTEXT) and maps it to LoGeR's existing
v67 source-attention lowstuff mask family.  The candidate and controls all use
beta_frame=0 so the only intended action is context-source K-side/logit
suppression in frame attention.
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
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_v68_phaseD_read_smoke import _eval_run, _load_kitti_gt  # noqa: E402


ROOT = Path("results/acl2_v96tf_vggt4d_informed_semantic_gauge_preserving_memory_control")
OUT_ROOT = ROOT / "trackJ_read_skip_pilot"
CASE_ATLAS = ROOT / "trackA_case_response_atlas/rows.csv"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
DATA_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/sequences")
GT_ROOT = Path("/mnt/data/users/chengshun.wang/data/kitti_odometry/dataset/poses")
CHECKPOINT = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/latest.pt")
CONFIG = Path("/mnt/data/users/chengshun.wang/pjs/LoGeR/ckpts/LoGeR/original_config.yaml")
TRACKH_READ21_GATE_CUE = "gate.gg_qk_shallow_mean_le_q40.chunk_ge_6.rtok_ge_0p005.then_v78_l07"

VARIANTS: dict[str, dict[str, Any]] = {
    "READ_NO_ACTION": {
        "role": "baseline",
        "mask": "",
        "semantic_contract": "beta=0 read_path frame baseline; no context source skip",
    },
    "READ_SKIP_WEAK_CONTEXT_K": {
        "role": "candidate",
        "mask": "v67_source_attention_lowstuff_q90",
        "semantic_contract": "J3 READ WEAK_SCALE_CONTEXT mapped to v67 lowstuff source-attention q90 early K/logit suppression",
    },
    "READ_RANDOM_SAME_MASS_SKIP": {
        "role": "random_same_mass_control",
        "mask": "v67_source_attention_lowstuff_q90_random_same_mass",
        "semantic_contract": "same selected-token mass as lowstuff source attention, deterministic random locations",
    },
    "READ_SEMANTIC_ROTATION_SKIP": {
        "role": "semantic_rotation_control",
        "mask": "v67_source_attention_lowstuff_q90_shuffled",
        "semantic_contract": "same source-attention lowstuff rule after deterministic semantic-label shuffle",
    },
    "READ_DG_Q90_HEAD_SKIP": {
        "role": "candidate_per_head_carrier",
        "mask": "dg_q90",
        "semantic_contract": "TrackD/H candidate D: DG q90 source eligibility restricted by selected layer/head carrier",
    },
    "READ_DG_Q90_RANDOM_SAME_MASS_SKIP": {
        "role": "random_same_mass_per_head_carrier_control",
        "mask": "dg_q90_random_same_mass",
        "semantic_contract": "same eligible-token mass as DG q90 with deterministic random token locations, restricted by same layer/head scope",
    },
    "READ_DG_Q90_HEAD_ANCHOR_RESCUE": {
        "role": "candidate_per_head_carrier_anchor_rescue",
        "mask": "dg_q90_anchor_rescue",
        "semantic_contract": (
            "TrackD/H Candidate B+D: DG q90 per-head source eligibility plus stable-anchor source-bias compensation"
        ),
    },
    "READ_DG_Q90_ANCHOR_RESCUE_RANDOM_SAME_MASS": {
        "role": "random_same_mass_per_head_carrier_anchor_rescue_control",
        "mask": "dg_q90_anchor_rescue_random_same_mass",
        "semantic_contract": (
            "same eligible-token mass as DG q90 anchor-rescue with deterministic random risk tokens and the same stable-anchor boost"
        ),
    },
    "READ21_GATED_L07_FIXED": {
        "role": "trackh_l07_fixed_diagnostic",
        "action_family": "frame_read",
        "read_cue": TRACKH_READ21_GATE_CUE,
        "beta_frame": 0.5,
        "beta_policy": "fixed",
        "beta_energy_target": 0.0,
        "semantic_contract": (
            "v95 READ21 runtime gate plus old L07 frame READ body at fixed beta=0.5; diagnostic comparator, not gauge normalized"
        ),
    },
    "READ21_GATED_L07_GAUGE_NORM_T030": {
        "role": "candidate_gauge_normalized_l07",
        "action_family": "frame_read",
        "read_cue": TRACKH_READ21_GATE_CUE,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": (
            "Candidate C: v95 READ21 runtime gate plus old L07 body with internal frame-bias energy normalization"
        ),
    },
    "READ21_GATED_L07_GAUGE_NORM_T045": {
        "role": "candidate_gauge_normalized_l07_t045",
        "action_family": "frame_read",
        "read_cue": TRACKH_READ21_GATE_CUE,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": (
            "Candidate C sensitivity: same runtime gate and old L07 body with a less conservative internal energy cap"
        ),
    },
    "READ21_GATED_L07_GAUGE_NORM_T050": {
        "role": "candidate_gauge_normalized_l07_t050",
        "action_family": "frame_read",
        "read_cue": TRACKH_READ21_GATE_CUE,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": (
            "Candidate C sensitivity: internal energy cap below observed active raw energy, used as final bounded cap check"
        ),
    },
    "READ21_GATED_L07_GEOMETRY_CONTROL_T030": {
        "role": "gated_l07_geometry_control",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.geometry_only",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and energy normalization, but old L07 semantic cue is replaced by geometry-only control",
    },
    "READ21_GATED_L07_LABEL_SHUFFLE_T030": {
        "role": "gated_l07_label_shuffle_control",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and energy normalization, but old L07 semantic labels are deterministically shuffled",
    },
    "READ21_GATED_L07_CONFIDENCE_SHUFFLE_T030": {
        "role": "gated_l07_confidence_shuffle_control",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and energy normalization, but old L07 semantic confidence is deterministically shuffled",
    },
    "READ21_GATED_L07_SAME_MASS_RANDOM_T030": {
        "role": "gated_l07_same_mass_random_control",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and energy normalization, but old L07 read mass is randomized per frame",
    },
    "READ21_GATED_L07_GROUP_RANDOM_T030": {
        "role": "gated_l07_group_random_control",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and energy normalization, but old L07 cue values are randomized within semantic groups",
    },
    "READ21_GATED_L07_GEOMETRY_CONTROL_T045": {
        "role": "gated_l07_geometry_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.geometry_only",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T045 energy normalization, but old L07 semantic cue is replaced by geometry-only control",
    },
    "READ21_GATED_L07_LABEL_SHUFFLE_T045": {
        "role": "gated_l07_label_shuffle_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T045 energy normalization, but old L07 semantic labels are deterministically shuffled",
    },
    "READ21_GATED_L07_CONFIDENCE_SHUFFLE_T045": {
        "role": "gated_l07_confidence_shuffle_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T045 energy normalization, but old L07 semantic confidence is deterministically shuffled",
    },
    "READ21_GATED_L07_SAME_MASS_RANDOM_T045": {
        "role": "gated_l07_same_mass_random_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T045 energy normalization, but old L07 read mass is randomized per frame",
    },
    "READ21_GATED_L07_GROUP_RANDOM_T045": {
        "role": "gated_l07_group_random_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T045 energy normalization, but old L07 cue values are randomized within semantic groups",
    },
    "READ21_GATED_L07_CONFNEUTRAL_T030": {
        "role": "candidate_gauge_normalized_l07_confidence_neutral_t030",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "Lower-strength confidence-neutral L07 repair after T045 full global-gauge regression",
    },
    "READ21_GATED_L07_CONFNEUTRAL_LABEL_SHUFFLE_T030": {
        "role": "gated_l07_confidence_neutral_label_shuffle_control_t030",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T030 confidence-neutral L07 with labels/groups shuffled before risk/support construction",
    },
    "READ21_GATED_L07_CONFNEUTRAL_SAME_MASS_RANDOM_T030": {
        "role": "gated_l07_confidence_neutral_same_mass_random_control_t030",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T030 confidence-neutral L07 with per-frame same-mass randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_GROUP_RANDOM_T030": {
        "role": "gated_l07_confidence_neutral_group_random_control_t030",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.30,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T030 confidence-neutral L07 with group-stratified randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_T035": {
        "role": "candidate_gauge_normalized_l07_confidence_neutral_t035",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.35,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "Intermediate-strength confidence-neutral L07 repair between T030 mechanism miss and T045 full regression",
    },
    "READ21_GATED_L07_CONFNEUTRAL_LABEL_SHUFFLE_T035": {
        "role": "gated_l07_confidence_neutral_label_shuffle_control_t035",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.35,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T035 confidence-neutral L07 with labels/groups shuffled before risk/support construction",
    },
    "READ21_GATED_L07_CONFNEUTRAL_SAME_MASS_RANDOM_T035": {
        "role": "gated_l07_confidence_neutral_same_mass_random_control_t035",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.35,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T035 confidence-neutral L07 with per-frame same-mass randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_GROUP_RANDOM_T035": {
        "role": "gated_l07_confidence_neutral_group_random_control_t035",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.35,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T035 confidence-neutral L07 with group-stratified randomized output",
    },
    "READ21_GATED_L07_GEOMETRY_CONTROL_T035": {
        "role": "gated_l07_geometry_control_t035",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.geometry_only",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.35,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T035 energy normalization, but old L07 body uses geometry-only read cue",
    },
    "READ21_GATED_L07_CONFIDENCE_SHUFFLE_T035": {
        "role": "gated_l07_confidence_shuffle_control_t035",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.35,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T035 energy normalization, but old L07 semantic confidence is deterministically shuffled",
    },
    "READ21_GATED_L07_CONFNEUTRAL_T045": {
        "role": "candidate_gauge_normalized_l07_confidence_neutral_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "Candidate C split: READ21 gate plus old L07 body with semantic trust neutralized before risk/support construction",
    },
    "READ21_GATED_L07_CONFNEUTRAL_LABEL_SHUFFLE_T045": {
        "role": "gated_l07_confidence_neutral_label_shuffle_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same confidence-neutral L07 body with labels/groups shuffled before risk/support construction",
    },
    "READ21_GATED_L07_CONFNEUTRAL_SAME_MASS_RANDOM_T045": {
        "role": "gated_l07_confidence_neutral_same_mass_random_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same confidence-neutral L07 body with per-frame same-mass randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_GROUP_RANDOM_T045": {
        "role": "gated_l07_confidence_neutral_group_random_control_t045",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.45,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same confidence-neutral L07 body with group-stratified randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_T050": {
        "role": "candidate_gauge_normalized_l07_confidence_neutral_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "Higher-strength confidence-neutral L07 probe after T035/T045 late-only near-miss",
    },
    "READ_L07_CONFNEUTRAL_CARRIER_T050": {
        "role": "candidate_carrier_scoped_l07_confidence_neutral_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": (
            "Candidate D+C: ungated confidence-neutral old L07 body scoped by raw-QK carrier layer/head/query CLI"
        ),
    },
    "READ_L07_CONFNEUTRAL_CARRIER_LABEL_SHUFFLE_T050": {
        "role": "carrier_scoped_l07_confidence_neutral_label_shuffle_control_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral_label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": (
            "same ungated carrier-scoped L07 body with labels/groups shuffled before support/risk construction"
        ),
    },
    "READ_L07_CONFNEUTRAL_CARRIER_SAME_MASS_RANDOM_T050": {
        "role": "carrier_scoped_l07_confidence_neutral_same_mass_random_control_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral_same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same ungated carrier-scoped L07 body with per-frame same-mass randomized output",
    },
    "READ_L07_CONFNEUTRAL_CARRIER_GROUP_RANDOM_T050": {
        "role": "carrier_scoped_l07_confidence_neutral_group_random_control_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral_group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same ungated carrier-scoped L07 body with group-stratified randomized output",
    },
    "READ_QKPAIR_KEYSTAB_CARRIER_T050": {
        "role": "candidate_carrier_scoped_qk_pair_key_stability_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral",
        "frame_bias_mode": "qk_pair_key_stability",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": (
            "Candidate D source-target repair: risky query tokens are paired with key-stability tokens in the selected raw-QK carrier head/query scope"
        ),
    },
    "READ_QKPAIR_KEYSTAB_CARRIER_LABEL_SHUFFLE_T050": {
        "role": "carrier_scoped_qk_pair_key_stability_label_shuffle_control_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral_label_shuffled",
        "frame_bias_mode": "qk_pair_key_stability",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same carrier-scoped QK-pair key-stability body with labels/groups shuffled before support/risk construction",
    },
    "READ_QKPAIR_KEYSTAB_CARRIER_PAIR_RANDOM_T050": {
        "role": "carrier_scoped_qk_pair_key_stability_pair_random_control_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral",
        "frame_bias_mode": "qk_pair_key_stability_random_same_mass",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same risky-query and key-stability inputs but QK pair locations are deterministically randomized with same mass",
    },
    "READ_QKPAIR_KEYSTAB_CARRIER_GROUP_RANDOM_T050": {
        "role": "carrier_scoped_qk_pair_key_stability_group_random_control_t050",
        "action_family": "frame_read",
        "read_cue": "v78.l07_l13.l07_action_only.confidence_neutral_group_stratified_random",
        "frame_bias_mode": "qk_pair_key_stability",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same carrier-scoped QK-pair key-stability body with group-stratified randomized risk output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_LABEL_SHUFFLE_T050": {
        "role": "gated_l07_confidence_neutral_label_shuffle_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T050 confidence-neutral L07 with labels/groups shuffled before risk/support construction",
    },
    "READ21_GATED_L07_CONFNEUTRAL_SAME_MASS_RANDOM_T050": {
        "role": "gated_l07_confidence_neutral_same_mass_random_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T050 confidence-neutral L07 with per-frame same-mass randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_GROUP_RANDOM_T050": {
        "role": "gated_l07_confidence_neutral_group_random_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T050 confidence-neutral L07 with group-stratified randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_T050": {
        "role": "candidate_gauge_normalized_l07_confidence_neutral_anchor_comp_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_anchor_comp",
        "mask": "semantic_anchor_bank",
        "context_source_skip_impl": "anchor_boost",
        "context_source_skip_mode": "boost",
        "context_source_skip_soft_rho": 0.10,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "Candidate B+C: confidence-neutral L07 with stable-anchor/layout risk compensation",
    },
    "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_LABEL_SHUFFLE_T050": {
        "role": "gated_l07_confidence_neutral_anchor_comp_label_shuffle_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_anchor_comp_label_shuffled",
        "mask": "semantic_anchor_bank",
        "context_source_skip_impl": "anchor_boost",
        "context_source_skip_mode": "boost",
        "context_source_skip_soft_rho": 0.10,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T050 confidence-neutral anchor-comp L07 with labels/groups shuffled before support/risk construction",
    },
    "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_SAME_MASS_RANDOM_T050": {
        "role": "gated_l07_confidence_neutral_anchor_comp_same_mass_random_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_anchor_comp_same_attention_mass_random",
        "mask": "semantic_anchor_bank",
        "context_source_skip_impl": "anchor_boost",
        "context_source_skip_mode": "boost",
        "context_source_skip_soft_rho": 0.10,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T050 confidence-neutral anchor-comp L07 with per-frame same-mass randomized output",
    },
    "READ21_GATED_L07_CONFNEUTRAL_ANCHORCOMP_GROUP_RANDOM_T050": {
        "role": "gated_l07_confidence_neutral_anchor_comp_group_random_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_neutral_anchor_comp_group_stratified_random",
        "mask": "semantic_anchor_bank",
        "context_source_skip_impl": "anchor_boost",
        "context_source_skip_mode": "boost",
        "context_source_skip_soft_rho": 0.10,
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "T050 confidence-neutral anchor-comp L07 with group-stratified randomized output",
    },
    "READ21_GATED_L07_GEOMETRY_CONTROL_T050": {
        "role": "gated_l07_geometry_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.geometry_only",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T050 energy normalization, but old L07 semantic cue is replaced by geometry-only control",
    },
    "READ21_GATED_L07_LABEL_SHUFFLE_T050": {
        "role": "gated_l07_label_shuffle_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.label_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T050 energy normalization, but old L07 semantic labels are deterministically shuffled",
    },
    "READ21_GATED_L07_CONFIDENCE_SHUFFLE_T050": {
        "role": "gated_l07_confidence_shuffle_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.confidence_shuffled",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T050 energy normalization, but old L07 semantic confidence is deterministically shuffled",
    },
    "READ21_GATED_L07_SAME_MASS_RANDOM_T050": {
        "role": "gated_l07_same_mass_random_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.same_attention_mass_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T050 energy normalization, but old L07 read mass is randomized per frame",
    },
    "READ21_GATED_L07_GROUP_RANDOM_T050": {
        "role": "gated_l07_group_random_control_t050",
        "action_family": "frame_read",
        "read_cue": f"{TRACKH_READ21_GATE_CUE}.group_stratified_random",
        "beta_frame": 0.5,
        "beta_policy": "bias_energy_norm",
        "beta_energy_target": 0.50,
        "beta_min": 0.15,
        "beta_max": 1.0,
        "semantic_contract": "same READ21 runtime gate and T050 energy normalization, but old L07 cue values are randomized within semantic groups",
    },
    "READ_STABLE_ANCHOR_RESCUE": {
        "role": "candidate_anchor_compensation",
        "mask": "v96_source_attention_lowstuff_q90_anchor_rescue",
        "semantic_contract": (
            "J4 fail-forward: suppress weak-context lowstuff source-attention q90 while mildly boosting semantic-anchor source mass"
        ),
    },
    "READ_STABLE_ANCHOR_RESCUE_RANDOM_RISK": {
        "role": "random_same_mass_anchor_compensation_control",
        "mask": "v96_source_attention_lowstuff_q90_anchor_rescue_random_same_mass",
        "semantic_contract": "same stable-anchor boost with same-mass random weak-context risk suppression",
    },
    "READ_STABLE_ANCHOR_RESCUE_SEMANTIC_ROTATION_RISK": {
        "role": "semantic_rotation_anchor_compensation_control",
        "mask": "v96_source_attention_lowstuff_q90_anchor_rescue_shuffled",
        "semantic_contract": "same stable-anchor boost with semantic-label-rotated weak-context risk suppression",
    },
}

LOWER_IS_BETTER = [
    "local_sim3_ate_rmse_m",
    "head10_to_tail10_pose_sim3_rmse_m",
    "scale_cv_head_mid_tail_pose_sim3",
]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple)):
                    clean[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
                else:
                    clean[key] = value
            writer.writerow(clean)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def median(values: list[float]) -> float | None:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(np.asarray(xs, dtype=float))) if xs else None


def ratio_improvement(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    return float((float(base) - float(cand)) / max(abs(float(base)), 1e-12))


def select_cases(max_bad: int, max_good: int, *, case_selection: str, boundary_context: str) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(CASE_ATLAS.open(newline="", encoding="utf-8")))
    if case_selection == "swa_tracke":
        swa_rows = [
            row for row in rows
            if "SWA_HANDOFF_CANDIDATE" in str(row.get("action_response_labels", ""))
        ]
        good = [
            row for row in swa_rows
            if "GOOD_PROTECTION" in str(row.get("action_response_labels", ""))
            or str(row.get("case_label_offline_only", "")).strip().lower() == "good"
        ]
        bad = [row for row in swa_rows if row not in good]

        def swa_sort_key(row: dict[str, Any]) -> tuple[str, float]:
            l3 = finite(row.get("L3_handoff_transfer_penalty_proxy"))
            return (str(row.get("seq", "")), -(l3 or 0.0))

        selected_bad = sorted(bad, key=swa_sort_key)[:max_bad]
        selected_good = sorted(good, key=swa_sort_key)[:max_good]
        out: list[dict[str, Any]] = []
        for row in selected_bad:
            out.append(case_from_atlas(row, "SWA_HANDOFF_NON_GOOD", boundary_context=boundary_context))
        for row in selected_good:
            out.append(case_from_atlas(row, "SWA_HANDOFF_GOOD_CONTROL", boundary_context=boundary_context))
        if not out:
            raise ValueError(f"no SWA TrackE cases selected from {CASE_ATLAS}")
        return out

    bad = [row for row in rows if "READ_LOCAL_BAD" in str(row.get("action_response_labels", ""))]
    good = [
        row for row in rows
        if "GOOD_PROTECTION" in str(row.get("action_response_labels", ""))
        and "READ_LOCAL_BAD" not in str(row.get("action_response_labels", ""))
    ]

    def sort_key(row: dict[str, Any]) -> tuple[str, float]:
        l1 = finite(row.get("L1_local_sim3_ate"))
        return (str(row.get("seq", "")), -(l1 or 0.0))

    bad = sorted(bad, key=sort_key)
    selected_bad: list[dict[str, Any]] = []
    seen_seq: set[str] = set()
    for row in bad:
        seq = str(row.get("seq", ""))
        if seq not in seen_seq:
            selected_bad.append(row)
            seen_seq.add(seq)
        if len(selected_bad) >= max_bad:
            break
    for row in bad:
        if len(selected_bad) >= max_bad:
            break
        if row not in selected_bad:
            selected_bad.append(row)

    good = sorted(good, key=sort_key)
    selected_good = good[:max_good]

    out: list[dict[str, Any]] = []
    for row in selected_bad:
        out.append(case_from_atlas(row, "READ_LOCAL_BAD", boundary_context=boundary_context))
    for row in selected_good:
        out.append(case_from_atlas(row, "GOOD_PROTECTION", boundary_context=boundary_context))
    if not out:
        raise ValueError(f"no J4 cases selected from {CASE_ATLAS}")
    return out


def case_from_atlas(row: dict[str, Any], bucket: str, *, boundary_context: str) -> dict[str, Any]:
    seq = f"{int(str(row['seq'])):02d}"
    prev_chunk = int(row["prev_chunk"])
    curr_chunk = int(row["curr_chunk"])
    if boundary_context == "prev_curr":
        run_chunk = max(prev_chunk, 0)
        start = max(run_chunk * 29, 0)
        end = max(curr_chunk * 29 + 32, start + 32)
    else:
        run_chunk = curr_chunk
        start = max(curr_chunk * 29, 0)
        end = start + 32
    return {
        "case_id": str(row["case_id"]),
        "seq": seq,
        "chunk": curr_chunk,
        "prev_chunk": prev_chunk,
        "curr_chunk": curr_chunk,
        "run_chunk_offset": run_chunk,
        "boundary_context": boundary_context,
        "start_frame": start,
        "end_frame": end,
        "bucket": bucket,
        "atlas_L1_local_sim3_ate": row.get("L1_local_sim3_ate"),
        "atlas_L2_head_tail_proxy_error": row.get("L2_head_tail_proxy_error"),
        "atlas_L3_handoff_transfer_penalty_proxy": row.get("L3_handoff_transfer_penalty_proxy"),
        "atlas_L3_J_handoff": row.get("L3_J_handoff"),
        "atlas_v96_recommended_next_track": row.get("v96_recommended_next_track"),
        "atlas_action_response_labels": row.get("action_response_labels"),
    }


def build_cmd(case: dict[str, Any], variant: str, out_dir: Path, args: argparse.Namespace) -> list[str]:
    seq = str(case["seq"])
    cmd = [
        str(CONDA),
        "run",
        "--no-capture-output",
        "-n",
        "loger",
        "python",
        "run_pipeline_abc_v2.py",
        "--input",
        str(DATA_ROOT / seq / "image_2"),
        "--output_video",
        "",
        "--output_txt",
        str(out_dir / f"{seq}.txt"),
        "--checkpoint",
        str(CHECKPOINT),
        "--config",
        str(CONFIG),
        "--chunk_size",
        "32",
        "--chunk_overlap",
        "3",
        "--start_frame",
        str(int(case["start_frame"])),
        "--end_frame",
        str(int(case["end_frame"])),
        "--global_chunk_offset",
        str(int(case.get("run_chunk_offset", case["chunk"]))),
        "--device",
        "cuda",
        "--hybrid_memory_mode",
        "read_path_only",
        "--hmc_commit_mode",
        "controlled",
        "--semantic_prior_mode",
        "spg_v2",
        "--stage_c_cache_mode",
        "read",
        "--stage_c_cache_dir",
        str(ROOT.parent / f"kitti_preprocess/{seq}/stage_c_cache_semantic_chunks"),
        "--stage_c_cache_require_hit",
        "1",
    ]
    variant_def = VARIANTS[variant]
    if str(variant_def.get("action_family", "")) == "frame_read":
        cmd += [
            "--enable_frame_read_control",
            "1",
            "--read_path",
            "frame",
            "--beta_frame",
            str(variant_def.get("beta_frame", 0.5)),
            "--frame_bias_mode",
            str(variant_def.get("frame_bias_mode", "key")),
            "--frame_bias_query_region",
            str(args.frame_bias_query_region),
            "--frame_bias_head_indices",
            str(args.frame_bias_head_indices),
            "--read_layer_mode",
            str(args.read_layer_mode),
            "--read_single_layer",
            str(args.read_single_layer),
            "--read_cue_source",
            str(variant_def["read_cue"]),
            "--read_topk_frac",
            str(variant_def.get("read_topk_frac", 0.1)),
            "--read_target_mass",
            str(variant_def.get("read_target_mass", 0.1)),
            "--read_calib_tau",
            str(variant_def.get("read_calib_tau", 0.05)),
            "--read_blend_lambda",
            str(variant_def.get("read_blend_lambda", 0.5)),
            "--beta_policy",
            str(variant_def.get("beta_policy", "fixed")),
            "--beta_energy_target",
            str(variant_def.get("beta_energy_target", 0.0)),
            "--beta_min",
            str(variant_def.get("beta_min", 0.5)),
            "--beta_max",
            str(variant_def.get("beta_max", 1.5)),
            "--frame_attention_record_bias_mass",
            "1",
            "--frame_attention_bias_mass_max_queries",
            str(args.attention_mass_max_queries),
            "--fast_cue_eval",
            "1",
            "--empty_cuda_cache_each_chunk",
            "1",
            "--hybrid_debug_jsonl",
            str(out_dir / "hmc_state_hash.jsonl"),
            "--read_cue_patch_dump_dir",
            str(out_dir / "read_cue_patch_dumps"),
            "--read_cue_patch_dump_dtype",
            "float16",
        ]
    else:
        cmd += [
        "--read_path",
        "frame",
        "--beta_frame",
        "0.0",
        "--frame_bias_mode",
        "key",
        "--read_cue_source",
        "v78.l07_l13.l07_action_only",
        "--fast_cue_eval",
        "1",
        "--empty_cuda_cache_each_chunk",
        "1",
        "--hybrid_debug_jsonl",
        str(out_dir / "hmc_state_hash.jsonl"),
        "--read_cue_patch_dump_dir",
        str(out_dir / "read_cue_patch_dumps"),
        "--read_cue_patch_dump_dtype",
        "float16",
        ]
    mask = str(VARIANTS[variant].get("mask", ""))
    if mask:
        cmd += [
            "--enable_context_source_skip",
            "1",
            "--context_source_skip_impl",
            str(variant_def.get("context_source_skip_impl", "bias")),
            "--context_source_skip_scope",
            str(variant_def.get("context_source_skip_scope", "frame")),
            "--context_source_skip_mode",
            str(variant_def.get("context_source_skip_mode", "soft")),
            "--context_source_skip_mask",
            mask,
            "--context_source_skip_frame_region",
            str(args.frame_region),
            "--context_source_skip_query_region",
            str(args.query_region),
            "--context_source_skip_soft_rho",
            str(args.soft_rho),
            "--context_source_skip_soft_min_keep",
            str(args.soft_min_keep),
            "--context_source_skip_layer_mode",
            str(args.layer_mode),
            "--context_source_skip_single_layer",
            str(args.single_layer),
            "--context_source_skip_head_indices",
            str(args.head_indices),
            "--context_source_skip_record_attention_mass",
            "1",
            "--context_source_skip_attention_mass_max_queries",
            str(args.attention_mass_max_queries),
            "--semantic_anchor_mode",
            str(args.semantic_anchor_mode),
            "--semantic_anchor_target_ratio",
            str(args.semantic_anchor_target_ratio),
            "--semantic_anchor_min_ratio",
            str(args.semantic_anchor_min_ratio),
            "--semantic_anchor_max_ratio",
            str(args.semantic_anchor_max_ratio),
            "--semantic_anchor_min_score",
            str(args.semantic_anchor_min_score),
            "--semantic_anchor_missing_trust_policy",
            str(args.semantic_anchor_missing_trust_policy),
            "--semantic_anchor_value_fallback",
            str(args.semantic_anchor_value_fallback),
        ]
        if "context_source_skip_soft_rho" in variant_def:
            cmd[cmd.index("--context_source_skip_soft_rho") + 1] = str(variant_def["context_source_skip_soft_rho"])
        if "context_source_skip_soft_min_keep" in variant_def:
            cmd[cmd.index("--context_source_skip_soft_min_keep") + 1] = str(variant_def["context_source_skip_soft_min_keep"])
    trace_dir_text = str(getattr(args, "swa_raw_transport_trace_dir", "") or "").strip()
    if trace_dir_text:
        trace_dir = Path(trace_dir_text)
        if not trace_dir.is_absolute():
            trace_dir = out_dir / trace_dir
        cmd += [
            "--swa_raw_transport_trace_dir",
            str(trace_dir),
            "--swa_raw_transport_trace_layer_mode",
            str(args.swa_raw_transport_trace_layer_mode),
            "--swa_raw_transport_trace_single_layer",
            str(args.swa_raw_transport_trace_single_layer),
            "--swa_raw_transport_trace_max_queries",
            str(args.swa_raw_transport_trace_max_queries),
            "--swa_raw_transport_trace_topk",
            str(args.swa_raw_transport_trace_topk),
            "--swa_raw_transport_trace_direct_match_only",
            str(int(getattr(args, "swa_raw_transport_trace_direct_match_only", 0))),
            "--swa_raw_transport_trace_query_block_size",
            str(int(getattr(args, "swa_raw_transport_trace_query_block_size", 128))),
        ]
    if int(getattr(args, "v68_export_full_pca_debug", 0)):
        pca_subdir = str(getattr(args, "v68_layer_pca_feature_subdir", "pca_features") or "pca_features")
        cmd += [
            "--v68_export_full_pca_debug",
            "1",
            "--v68_layer_pca_feature_dir",
            str(out_dir / pca_subdir),
            "--v68_pca_taps",
            str(args.v68_pca_taps),
            "--v68_pca_layers",
            str(args.v68_pca_layers),
            "--v68_pca_max_feature_dim",
            str(args.v68_pca_max_feature_dim),
        ]
    return cmd


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    log_path = out_dir / "run.log"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(job["cmd"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    job.update(
        {
            "returncode": int(proc.returncode),
            "duration_sec": float(time.time() - started),
            "run_log": str(log_path),
            "output_txt": str(out_dir / f"{job['seq']}.txt"),
            "hmc_jsonl": str(out_dir / "hmc_state_hash.jsonl"),
        }
    )
    write_json(out_dir / "job_summary.json", job)
    return job


def build_jobs(cases: list[dict[str, Any]], variants: list[str], gpus: list[int], args: argparse.Namespace) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for case in cases:
        for variant in variants:
            out_dir = OUT_ROOT / str(case["case_id"]) / variant
            cmd = build_cmd(case, variant, out_dir, args)
            skipped = bool(args.skip_existing and (out_dir / f"{case['seq']}.txt").exists() and (out_dir / "hmc_state_hash.jsonl").exists())
            jobs.append(
                {
                    "job_index": len(jobs),
                    "case_id": case["case_id"],
                    "seq": case["seq"],
                    "chunk": case["chunk"],
                    "prev_chunk": case.get("prev_chunk"),
                    "curr_chunk": case.get("curr_chunk"),
                    "run_chunk_offset": case.get("run_chunk_offset"),
                    "boundary_context": case.get("boundary_context"),
                    "bucket": case["bucket"],
                    "variant": variant,
                    "variant_role": VARIANTS[variant]["role"],
                    "semantic_contract": VARIANTS[variant]["semantic_contract"],
                    "start_frame": case["start_frame"],
                    "end_frame": case["end_frame"],
                    "atlas_L1_local_sim3_ate": case["atlas_L1_local_sim3_ate"],
                    "atlas_L2_head_tail_proxy_error": case["atlas_L2_head_tail_proxy_error"],
                    "atlas_L3_handoff_transfer_penalty_proxy": case.get("atlas_L3_handoff_transfer_penalty_proxy"),
                    "atlas_L3_J_handoff": case.get("atlas_L3_J_handoff"),
                    "atlas_v96_recommended_next_track": case.get("atlas_v96_recommended_next_track"),
                    "atlas_action_response_labels": case["atlas_action_response_labels"],
                    "gpu": int(gpus[len(jobs) % len(gpus)]),
                    "out_dir": str(out_dir),
                    "cmd": cmd,
                    "cmd_shell": shlex.join(cmd),
                    "skipped": skipped,
                    "returncode": 0 if skipped else None,
                }
            )
    return jobs


def run_jobs(jobs: list[dict[str, Any]], gpus: list[int]) -> list[dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {int(job["job_index"]): dict(job) for job in jobs if job.get("skipped")}
    queues: dict[int, list[dict[str, Any]]] = {gpu: [] for gpu in gpus}
    for job in jobs:
        if not job.get("skipped"):
            queues.setdefault(int(job["gpu"]), []).append(dict(job))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(queues), 1)) as pool:
        futures: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
        for gpu, queue in queues.items():
            if queue:
                futures[pool.submit(run_job, queue.pop(0))] = int(gpu)
        while futures:
            for future in concurrent.futures.as_completed(list(futures)):
                gpu = futures.pop(future)
                break
            result = future.result()
            completed[int(result["job_index"])] = result
            print(
                "done",
                f"case={result['case_id']}",
                f"variant={result['variant']}",
                f"gpu={result['gpu']}",
                f"returncode={result['returncode']}",
                f"duration_sec={result['duration_sec']:.1f}",
                flush=True,
            )
            if queues.get(gpu):
                futures[pool.submit(run_job, queues[gpu].pop(0))] = gpu
    return [completed.get(int(job["job_index"]), job) for job in jobs]


def extract_trace_stats(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "hmc_state_hash.jsonl")
    if not rows:
        return {"hmc_debug_rows": 0}
    frame_rows: list[dict[str, Any]] = []
    swa_rows: list[dict[str, Any]] = []
    for row in rows:
        frame = row.get("control_trace", {}).get("hook_effect_summary", {}).get("frame_attention", {})
        if isinstance(frame, dict):
            frame_rows.append(frame)
        swa = row.get("control_trace", {}).get("hook_effect_summary", {}).get("swa_read", {})
        if isinstance(swa, dict):
            swa_rows.append(swa)

    def mean_key(key: str, source_rows: list[dict[str, Any]] | None = None) -> float | None:
        vals = [finite(row.get(key)) for row in (source_rows or frame_rows)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(np.asarray(vals, dtype=float))) if vals else None

    def median_row_key(key: str) -> float | None:
        vals = [finite(row.get(key)) for row in rows]
        vals = [v for v in vals if v is not None]
        return median([float(v) for v in vals])

    def frac_row_bool(key: str) -> float | None:
        vals = [row.get(key) for row in rows if key in row]
        if not vals:
            return None
        return float(np.mean(np.asarray([1.0 if bool(v) else 0.0 for v in vals], dtype=float)))

    return {
        "hmc_debug_rows": len(rows),
        "trace_frame_attention_summary_rows": len(frame_rows),
        "trace_swa_read_summary_rows": len(swa_rows),
        "trace_num_context_source_skip_applied": sum(int(row.get("num_context_source_skip_applied", 0) or 0) for row in frame_rows),
        "trace_mean_abs_bias": mean_key("mean_abs_bias"),
        "trace_max_abs_bias": max([finite(row.get("max_abs_bias")) or 0.0 for row in frame_rows], default=0.0),
        "trace_mean_attention_mass_removed_before": mean_key("mean_attention_mass_removed_before"),
        "trace_mean_attention_mass_removed_after": mean_key("mean_attention_mass_removed_after"),
        "trace_mean_attention_mass_retained_before": mean_key("mean_attention_mass_retained_before"),
        "trace_mean_attention_mass_retained_after": mean_key("mean_attention_mass_retained_after"),
        "trace_mean_attention_mass_stable_anchor_before": mean_key("mean_attention_mass_stable_anchor_before"),
        "trace_mean_attention_mass_stable_anchor_after": mean_key("mean_attention_mass_stable_anchor_after"),
        "trace_mean_attention_mass_stable_anchor_actual_after": mean_key("mean_attention_mass_stable_anchor_actual_after"),
        "trace_mean_attention_mass_stable_anchor_preservation_ratio": mean_key("mean_attention_mass_stable_anchor_preservation_ratio"),
        "trace_mean_attention_mass_stable_anchor_tokens": mean_key("mean_attention_mass_stable_anchor_tokens"),
        "trace_mean_source_weight_min": mean_key("mean_source_weight_min"),
        "trace_mean_source_weight_mean": mean_key("mean_source_weight_mean"),
        "trace_mean_context_source_keep_ratio": mean_key("mean_context_source_keep_ratio"),
        "trace_mean_selected_token_ratio": mean_key("mean_context_source_selected_token_ratio"),
        "trace_mean_selected_group_lowstuff_frac": mean_key("mean_context_source_selected_group_lowstuff_frac"),
        "trace_mean_selected_group_structure_frac": mean_key("mean_context_source_selected_group_structure_frac"),
        "trace_mean_selected_group_movable_frac": mean_key("mean_context_source_selected_group_movable_frac"),
        "trace_mean_selected_fine_sky_frac": mean_key("mean_context_source_selected_fine_sky_frac"),
        "trace_mean_semantic_anchor_rescue_source_tokens": mean_key("mean_semantic_anchor_rescue_source_tokens"),
        "trace_mean_semantic_anchor_rescue_source_ratio": mean_key("mean_semantic_anchor_rescue_source_ratio"),
        "trace_mean_semantic_anchor_rescue_source_score_mean": mean_key("mean_semantic_anchor_rescue_source_score_mean"),
        "trace_frac_semantic_anchor_rescue_source_available": mean_key("frac_semantic_anchor_rescue_source_available"),
        "trace_mean_frame_bias_positive_pair_mass_before": mean_key("mean_frame_bias_positive_pair_mass_before"),
        "trace_mean_frame_bias_positive_pair_mass_after": mean_key("mean_frame_bias_positive_pair_mass_after"),
        "trace_mean_frame_bias_positive_pair_mass_lift": mean_key("mean_frame_bias_positive_pair_mass_lift"),
        "trace_mean_frame_bias_negative_pair_mass_before": mean_key("mean_frame_bias_negative_pair_mass_before"),
        "trace_mean_frame_bias_negative_pair_mass_after": mean_key("mean_frame_bias_negative_pair_mass_after"),
        "trace_mean_frame_bias_negative_pair_mass_lift": mean_key("mean_frame_bias_negative_pair_mass_lift"),
        "trace_mean_frame_bias_positive_pair_fraction": mean_key("mean_frame_bias_positive_pair_fraction"),
        "trace_mean_frame_bias_negative_pair_fraction": mean_key("mean_frame_bias_negative_pair_fraction"),
        "trace_mean_frame_bias_positive_bias_mean": mean_key("mean_frame_bias_positive_bias_mean"),
        "trace_mean_frame_bias_negative_bias_mean": mean_key("mean_frame_bias_negative_bias_mean"),
        "trace_v95_gate_active_frac": frac_row_bool("prior_v95_trackH_gate_active"),
        "trace_v95_gate_strict_noop_frac": frac_row_bool("prior_v95_trackH_gate_strict_noop"),
        "trace_beta_frame_effective_median": median_row_key("prior_beta_frame_effective"),
        "trace_beta_raw_frame_bias_energy_median": median_row_key("prior_beta_raw_frame_bias_energy"),
        "trace_beta_energy_target_median": median_row_key("prior_beta_energy_target"),
        "trace_beta_was_clipped_frac": frac_row_bool("prior_beta_was_clipped"),
        "trace_swa_raw_transport_available_frac": mean_key("frac_swa_raw_transport_trace_available", swa_rows),
        "trace_swa_raw_transport_stable_nonempty_frac": mean_key("frac_swa_raw_transport_stable_groups_nonempty", swa_rows),
        "trace_swa_raw_transport_unreliable_nonempty_frac": mean_key("frac_swa_raw_transport_unreliable_groups_nonempty", swa_rows),
        "trace_swa_raw_transport_qk_similarity_mean": mean_key("mean_swa_raw_transport_qk_similarity_mean", swa_rows),
        "trace_swa_raw_transport_qk_similarity_max_mean": mean_key("mean_swa_raw_transport_qk_similarity_max_mean", swa_rows),
        "trace_swa_raw_transport_route_entropy_mean": mean_key("mean_swa_raw_transport_route_entropy_mean", swa_rows),
        "trace_swa_raw_transport_feature_residual_mean": mean_key("mean_swa_raw_transport_feature_residual_mean", swa_rows),
        "trace_swa_raw_transport_cache_k_stability_mean": mean_key("mean_swa_raw_transport_cache_k_stability_mean", swa_rows),
        "trace_swa_raw_transport_cache_v_stability_mean": mean_key("mean_swa_raw_transport_cache_v_stability_mean", swa_rows),
        "trace_swa_raw_transport_stable_pair_mass_mean": mean_key("mean_swa_raw_transport_stable_pair_mass_mean", swa_rows),
        "trace_swa_raw_transport_unreliable_pair_mass_mean": mean_key("mean_swa_raw_transport_unreliable_pair_mass_mean", swa_rows),
        "trace_swa_raw_transport_stable_actual_minus_random_mean": mean_key("mean_swa_raw_transport_stable_actual_minus_random_mean", swa_rows),
        "trace_swa_raw_transport_unreliable_actual_minus_random_mean": mean_key("mean_swa_raw_transport_unreliable_actual_minus_random_mean", swa_rows),
    }


def evaluate_jobs(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gt_cache: dict[str, tuple[Any, Any]] = {}
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for job in jobs:
        seq = str(job["seq"])
        run_dir = Path(job["out_dir"])
        if int(job.get("returncode") or 0) != 0:
            errors.append({**job, "eval_error": "job_returncode_nonzero"})
            continue
        try:
            if seq not in gt_cache:
                _, gt_poses_all, gt_pos_all = _load_kitti_gt(GT_ROOT / f"{seq}.txt")
                gt_cache[seq] = (gt_poses_all, gt_pos_all)
            gt_poses_all, gt_pos_all = gt_cache[seq]
            row = _eval_run(str(job["variant"]), run_dir, gt_poses_all, gt_pos_all, trajectory_name=f"{seq}.txt")
            row.update(
                {
                    "case_id": job["case_id"],
                    "seq": seq,
                    "chunk": int(job["chunk"]),
                    "prev_chunk": job.get("prev_chunk"),
                    "curr_chunk": job.get("curr_chunk"),
                    "run_chunk_offset": job.get("run_chunk_offset"),
                    "boundary_context": job.get("boundary_context"),
                    "bucket": job["bucket"],
                    "variant": job["variant"],
                    "variant_role": job["variant_role"],
                    "semantic_contract": job["semantic_contract"],
                    "start_frame": int(job["start_frame"]),
                    "end_frame": int(job["end_frame"]),
                    "run_returncode": int(job.get("returncode") or 0),
                }
            )
            row.update(extract_trace_stats(run_dir))
            retained_before = finite(row.get("trace_mean_attention_mass_retained_before"))
            retained_after = finite(row.get("trace_mean_attention_mass_retained_after"))
            stable_before = finite(row.get("trace_mean_attention_mass_stable_anchor_before"))
            stable_after = finite(row.get("trace_mean_attention_mass_stable_anchor_after"))
            row["stable_anchor_preservation_proxy"] = (
                float(stable_after / max(abs(stable_before), 1e-12))
                if stable_before is not None and stable_after is not None
                else (
                    float(retained_after / max(abs(retained_before), 1e-12))
                    if retained_before is not None and retained_after is not None else None
                )
            )
            row["stable_anchor_trace_type"] = (
                "true_stable_anchor_source_attention_mass"
                if stable_before is not None and stable_after is not None
                else "retained_source_attention_mass_proxy"
            )
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            errors.append({**job, "eval_error": f"{type(exc).__name__}: {exc}"})
    return rows, errors


def build_case_comparisons(
    rows: list[dict[str, Any]],
    *,
    candidate_variant: str,
    random_control_variant: str,
    semantic_control_variant: str,
    control_variants: list[str] | None = None,
) -> list[dict[str, Any]]:
    controls = list(dict.fromkeys(control_variants or [random_control_variant, semantic_control_variant]))
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), {})[str(row["variant"])] = row
    out: list[dict[str, Any]] = []
    for case_id, case_rows in sorted(by_case.items()):
        base = case_rows.get("READ_NO_ACTION")
        cand = case_rows.get(candidate_variant)
        if base is None or cand is None:
            continue
        for metric in LOWER_IS_BETTER:
            base_v = finite(base.get(metric))
            cand_v = finite(cand.get(metric))
            random_v = finite(case_rows.get(random_control_variant, {}).get(metric))
            rotate_v = finite(case_rows.get(semantic_control_variant, {}).get(metric))
            control_values = {
                control: finite(case_rows.get(control, {}).get(metric))
                for control in controls
            }
            control_margins = {
                control: ratio_improvement(value, cand_v)
                for control, value in control_values.items()
                if value is not None
            }
            finite_control_margins = [v for v in control_margins.values() if v is not None]
            out.append(
                {
                    "case_id": case_id,
                    "seq": cand.get("seq"),
                    "chunk": cand.get("chunk"),
                    "bucket": cand.get("bucket"),
                    "metric": metric,
                    "baseline": base_v,
                    "candidate": cand_v,
                    "random_same_mass": random_v,
                    "semantic_rotation": rotate_v,
                    "candidate_improvement_vs_baseline": ratio_improvement(base_v, cand_v),
                    "candidate_margin_vs_random": ratio_improvement(random_v, cand_v),
                    "candidate_margin_vs_semantic_rotation": ratio_improvement(rotate_v, cand_v),
                    "control_values": control_values,
                    "candidate_margins_vs_controls": control_margins,
                    "candidate_min_margin_vs_controls": min(finite_control_margins) if finite_control_margins else None,
                }
            )
    return out


def aggregate_decision(
    rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    *,
    candidate_variant: str,
    random_control_variant: str,
    semantic_control_variant: str,
    control_variants: list[str] | None = None,
) -> dict[str, Any]:
    controls = list(dict.fromkeys(control_variants or [random_control_variant, semantic_control_variant]))
    candidate_def = VARIANTS.get(candidate_variant, {})
    candidate_action_family = str(candidate_def.get("action_family", "context_source_skip"))
    bad_rows = [row for row in rows if row.get("bucket") == "READ_LOCAL_BAD"]
    good_rows = [row for row in rows if row.get("bucket") == "GOOD_PROTECTION"]

    def med_for(bucket: str, variant: str, metric: str) -> float | None:
        vals = [
            finite(row.get(metric))
            for row in rows
            if row.get("bucket") == bucket and row.get("variant") == variant
        ]
        return median([float(v) for v in vals if v is not None])

    metric_decisions: dict[str, Any] = {}
    bad_metric_passes: list[str] = []
    for metric in LOWER_IS_BETTER:
        bad_base = med_for("READ_LOCAL_BAD", "READ_NO_ACTION", metric)
        bad_cand = med_for("READ_LOCAL_BAD", candidate_variant, metric)
        bad_rand = med_for("READ_LOCAL_BAD", random_control_variant, metric)
        bad_rot = med_for("READ_LOCAL_BAD", semantic_control_variant, metric)
        control_medians = {control: med_for("READ_LOCAL_BAD", control, metric) for control in controls}
        good_base = med_for("GOOD_PROTECTION", "READ_NO_ACTION", metric)
        good_cand = med_for("GOOD_PROTECTION", candidate_variant, metric)
        bad_improve = ratio_improvement(bad_base, bad_cand)
        margin_random = ratio_improvement(bad_rand, bad_cand)
        margin_rotate = ratio_improvement(bad_rot, bad_cand)
        control_margins = {
            control: ratio_improvement(control_median, bad_cand)
            for control, control_median in control_medians.items()
            if control_median is not None
        }
        finite_control_margins = [v for v in control_margins.values() if v is not None]
        min_control_margin = min(finite_control_margins) if finite_control_margins else None
        controls_all_available = len(control_margins) == len(controls)
        controls_gate_pass = bool(
            controls_all_available
            and finite_control_margins
            and min_control_margin is not None
            and min_control_margin >= 0.05
        )
        good_worsen = None
        if good_base is not None and good_cand is not None:
            good_worsen = float((good_cand - good_base) / max(abs(good_base), 1e-12))
        key_pass = bool(
            bad_improve is not None
            and bad_improve >= 0.05
            and controls_gate_pass
        )
        if key_pass:
            bad_metric_passes.append(metric)
        metric_decisions[metric] = {
            "bad_baseline_median": bad_base,
            "bad_candidate_median": bad_cand,
            "bad_random_same_mass_median": bad_rand,
            "bad_semantic_rotation_median": bad_rot,
            "bad_required_control_medians": control_medians,
            "bad_improvement_vs_baseline": bad_improve,
            "candidate_margin_vs_random_same_mass": margin_random,
            "candidate_margin_vs_semantic_rotation": margin_rotate,
            "candidate_margins_vs_required_controls": control_margins,
            "candidate_min_margin_vs_required_controls": min_control_margin,
            "required_controls_all_available": controls_all_available,
            "good_baseline_median": good_base,
            "good_candidate_median": good_cand,
            "good_worsen_ratio": good_worsen,
            "bad_metric_gate_pass": key_pass,
        }

    cand_trace = [row for row in rows if row.get("variant") == candidate_variant]
    trace_delta_vals = []
    stable_proxy_vals = []
    frame_negative_lift_vals = []
    frame_bias_vals = []
    gate_active_vals = []
    beta_effective_vals = []
    raw_energy_vals = []
    active_beta_effective_vals = []
    active_raw_energy_vals = []
    for row in cand_trace:
        before = finite(row.get("trace_mean_attention_mass_removed_before"))
        after = finite(row.get("trace_mean_attention_mass_removed_after"))
        if before is not None and after is not None:
            trace_delta_vals.append(after - before)
        stable_proxy = finite(row.get("stable_anchor_preservation_proxy"))
        if stable_proxy is not None:
            stable_proxy_vals.append(stable_proxy)
        frame_negative_lift = finite(row.get("trace_mean_frame_bias_negative_pair_mass_lift"))
        if frame_negative_lift is not None:
            frame_negative_lift_vals.append(frame_negative_lift)
        bias = finite(row.get("trace_max_abs_bias"))
        if bias is not None:
            frame_bias_vals.append(bias)
        gate_active = finite(row.get("trace_v95_gate_active_frac"))
        if gate_active is not None:
            gate_active_vals.append(gate_active)
        beta_effective = finite(row.get("trace_beta_frame_effective_median"))
        if beta_effective is not None:
            beta_effective_vals.append(beta_effective)
        raw_energy = finite(row.get("trace_beta_raw_frame_bias_energy_median"))
        if raw_energy is not None:
            raw_energy_vals.append(raw_energy)
        if gate_active is not None and gate_active > 0.5:
            if beta_effective is not None:
                active_beta_effective_vals.append(beta_effective)
            if raw_energy is not None:
                active_raw_energy_vals.append(raw_energy)
    stable_trace_types = sorted({
        str(row.get("stable_anchor_trace_type"))
        for row in cand_trace
        if row.get("stable_anchor_trace_type")
    })
    true_stable_anchor_trace_available = "true_stable_anchor_source_attention_mass" in stable_trace_types
    if candidate_action_family == "frame_read":
        trace_fidelity_pass = bool(
            cand_trace
            and any(float(v) > 0.0 for v in gate_active_vals)
            and any(float(v) > 0.0 for v in frame_bias_vals)
            and frame_negative_lift_vals
            and median(frame_negative_lift_vals) is not None
            and float(median(frame_negative_lift_vals)) < 0.0
            and raw_energy_vals
            and any(float(v) > 0.0 for v in raw_energy_vals)
        )
    else:
        trace_fidelity_pass = bool(
            trace_delta_vals
            and median(trace_delta_vals) is not None
            and float(median(trace_delta_vals)) < 0.0
            and all(int(row.get("trace_num_context_source_skip_applied") or 0) > 0 for row in cand_trace)
        )
    stable_anchor_preservation_proxy = median(stable_proxy_vals)
    stable_anchor_proxy_pass = bool(
        true_stable_anchor_trace_available
        and stable_anchor_preservation_proxy is not None
        and stable_anchor_preservation_proxy >= 0.98
    )
    trace_beta_frame_effective_median = median(beta_effective_vals)
    trace_beta_raw_frame_bias_energy_median = median(raw_energy_vals)
    trace_beta_frame_effective_active_median = median(active_beta_effective_vals)
    trace_beta_raw_frame_bias_energy_active_median = median(active_raw_energy_vals)
    trace_frame_bias_negative_pair_mass_lift_median = median(frame_negative_lift_vals)
    trace_v95_gate_active_frac_median = median(gate_active_vals)
    global_safety_proxy_pass = stable_anchor_proxy_pass
    if candidate_action_family == "frame_read":
        beta_frame = finite(candidate_def.get("beta_frame"))
        global_safety_proxy_pass = bool(
            trace_beta_frame_effective_active_median is not None
            and beta_frame is not None
            and trace_beta_frame_effective_active_median <= beta_frame + 1e-12
            and trace_beta_raw_frame_bias_energy_active_median is not None
            and trace_beta_raw_frame_bias_energy_active_median > 0.0
        )

    good_j_base_vals = []
    good_j_cand_vals = []
    for case_id in sorted({str(row["case_id"]) for row in good_rows}):
        case_rows = {str(row["variant"]): row for row in rows if str(row["case_id"]) == case_id}
        base = case_rows.get("READ_NO_ACTION")
        cand = case_rows.get(candidate_variant)
        if base is None or cand is None:
            continue
        ratios = []
        for metric in LOWER_IS_BETTER:
            b = finite(base.get(metric))
            c = finite(cand.get(metric))
            if b is not None and c is not None:
                ratios.append(c / max(abs(b), 1e-12))
        if ratios:
            good_j_base_vals.append(1.0)
            good_j_cand_vals.append(float(np.mean(np.asarray(ratios, dtype=float))))
    good_j_worsen = None
    if good_j_cand_vals:
        good_j_worsen = float(median(good_j_cand_vals) - 1.0)
    good_safety_pass = bool(good_j_worsen is not None and good_j_worsen <= 0.02)

    mechanism_gate_pass = bool(
        not errors
        and bad_metric_passes
        and good_safety_pass
        and trace_fidelity_pass
        and global_safety_proxy_pass
        and stable_anchor_proxy_pass
    )
    return {
        "stage": "TrackJ_J4_read_weak_context_skip_pilot",
        "status": "complete",
        "gate_pass": mechanism_gate_pass,
        "runtime_action_allowed": False,
        "method_success": False,
        "full_method_success": False,
        "case_count": len({row["case_id"] for row in rows}),
        "metric_row_count": len(rows),
        "comparison_row_count": len(comparisons),
        "evaluation_error_count": len(errors),
        "candidate": candidate_variant,
        "candidate_action_family": candidate_action_family,
        "controls": controls,
        "bad_metric_passes": bad_metric_passes,
        "good_safety_pass": good_safety_pass,
        "good_j_short_proxy_worsen_ratio": good_j_worsen,
        "trace_fidelity_pass": trace_fidelity_pass,
        "trace_median_attention_mass_delta": median(trace_delta_vals),
        "trace_frame_bias_negative_pair_mass_lift_median": trace_frame_bias_negative_pair_mass_lift_median,
        "trace_v95_gate_active_frac_median": trace_v95_gate_active_frac_median,
        "trace_beta_frame_effective_median": trace_beta_frame_effective_median,
        "trace_beta_raw_frame_bias_energy_median": trace_beta_raw_frame_bias_energy_median,
        "trace_beta_frame_effective_active_median": trace_beta_frame_effective_active_median,
        "trace_beta_raw_frame_bias_energy_active_median": trace_beta_raw_frame_bias_energy_active_median,
        "global_safety_proxy_pass": global_safety_proxy_pass,
        "stable_anchor_preservation_proxy": stable_anchor_preservation_proxy,
        "stable_anchor_proxy_pass": stable_anchor_proxy_pass,
        "stable_anchor_trace_types": stable_trace_types,
        "true_stable_anchor_trace_available": true_stable_anchor_trace_available,
        "stable_anchor_proxy_note": (
            "true stable-anchor source-attention mass when available; retained-mass proxy otherwise"
        ),
        "metric_decisions": metric_decisions,
        "failure_reason": (
            "" if mechanism_gate_pass else
            "J4 mechanism gate failed or is unpromotable; see metric_decisions, trace_fidelity_pass, good_safety_pass, stable_anchor_proxy_pass."
        ),
        "gate_rule": (
            "bad READ_LOCAL median L1/L2/scale metric improves >=5%, candidate beats random and semantic-rotation controls "
            "by >=5%, good J_short proxy worsen <=2%, trace mass moves as expected, and stable-anchor preservation evidence >=0.98."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--max-bad-cases", type=int, default=4)
    parser.add_argument("--max-good-cases", type=int, default=4)
    parser.add_argument(
        "--case-selection",
        choices=("read_j4", "swa_tracke"),
        default="read_j4",
        help="Case bank selector. swa_tracke selects SWA_HANDOFF_CANDIDATE rows for Track E raw transport trace.",
    )
    parser.add_argument(
        "--boundary-context",
        choices=("current", "prev_curr"),
        default="current",
        help="current runs only curr_chunk; prev_curr starts at prev_chunk so target curr_chunk has SWA history.",
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--candidate-variant", default="READ_SKIP_WEAK_CONTEXT_K")
    parser.add_argument("--random-control-variant", default="READ_RANDOM_SAME_MASS_SKIP")
    parser.add_argument("--semantic-control-variant", default="READ_SEMANTIC_ROTATION_SKIP")
    parser.add_argument(
        "--required-control-variants",
        default="",
        help="Comma-separated controls that the candidate must beat by >=5%; defaults to random+semantic controls.",
    )
    parser.add_argument("--soft-rho", type=float, default=0.10)
    parser.add_argument("--soft-min-keep", type=float, default=0.90)
    parser.add_argument("--read-layer-mode", default="all")
    parser.add_argument("--read-single-layer", type=int, default=-1)
    parser.add_argument("--frame-bias-query-region", choices=("all", "head", "mid_tail", "tail"), default="all")
    parser.add_argument("--frame-bias-head-indices", default="")
    parser.add_argument("--layer-mode", default="early")
    parser.add_argument("--single-layer", type=int, default=-1)
    parser.add_argument("--head-indices", default="")
    parser.add_argument("--frame-region", choices=("all", "head", "mid_tail", "tail"), default="all")
    parser.add_argument("--query-region", choices=("all", "head", "mid_tail", "tail"), default="all")
    parser.add_argument("--attention-mass-max-queries", type=int, default=128)
    parser.add_argument("--swa-raw-transport-trace-dir", dest="swa_raw_transport_trace_dir", default="")
    parser.add_argument("--swa-raw-transport-trace-layer-mode", dest="swa_raw_transport_trace_layer_mode",
                        choices=("all", "first", "last", "single"), default="all")
    parser.add_argument("--swa-raw-transport-trace-single-layer", dest="swa_raw_transport_trace_single_layer",
                        type=int, default=-1)
    parser.add_argument("--swa-raw-transport-trace-max-queries", dest="swa_raw_transport_trace_max_queries",
                        type=int, default=128)
    parser.add_argument("--swa-raw-transport-trace-topk", dest="swa_raw_transport_trace_topk",
                        type=int, default=8)
    parser.add_argument("--swa-raw-transport-trace-direct-match-only",
                        dest="swa_raw_transport_trace_direct_match_only",
                        type=int, choices=(0, 1), default=0)
    parser.add_argument("--swa-raw-transport-trace-query-block-size",
                        dest="swa_raw_transport_trace_query_block_size",
                        type=int, default=128)
    parser.add_argument("--v68-export-full-pca-debug", dest="v68_export_full_pca_debug", type=int, default=0)
    parser.add_argument("--v68-layer-pca-feature-subdir", dest="v68_layer_pca_feature_subdir", default="pca_features")
    parser.add_argument(
        "--v68-pca-taps",
        dest="v68_pca_taps",
        default="pca_attn_global_k_layers,pca_attn_global_v_layers,pca_attn_frame_v_layers",
    )
    parser.add_argument("--v68-pca-layers", dest="v68_pca_layers", default="5,13,17")
    parser.add_argument("--v68-pca-max-feature-dim", dest="v68_pca_max_feature_dim", type=int, default=8)
    parser.add_argument("--semantic-anchor-mode", default="semantic")
    parser.add_argument("--semantic-anchor-target-ratio", type=float, default=0.12)
    parser.add_argument("--semantic-anchor-min-ratio", type=float, default=0.03)
    parser.add_argument("--semantic-anchor-max-ratio", type=float, default=0.30)
    parser.add_argument("--semantic-anchor-min-score", type=float, default=0.02)
    parser.add_argument("--semantic-anchor-missing-trust-policy", choices=("zero", "neutral"), default="zero")
    parser.add_argument("--semantic-anchor-value-fallback", choices=("off", "semantic_value"), default="off")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    global OUT_ROOT
    args = parse_args()
    OUT_ROOT = args.output_root
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in str(args.gpus).split(",") if x.strip()]
    variants = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    required_control_variants = [
        x.strip()
        for x in str(args.required_control_variants).split(",")
        if x.strip()
    ] or [args.random_control_variant, args.semantic_control_variant]
    if not gpus:
        raise ValueError("--gpus must list at least one GPU")
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; known={sorted(VARIANTS)}")
    for required in (
        args.candidate_variant,
        args.random_control_variant,
        args.semantic_control_variant,
        "READ_NO_ACTION",
        *required_control_variants,
    ):
        if required not in variants:
            raise ValueError(f"required variant {required} is not in --variants={variants}")
    cases = select_cases(
        args.max_bad_cases,
        args.max_good_cases,
        case_selection=str(args.case_selection),
        boundary_context=str(args.boundary_context),
    )
    jobs = build_jobs(cases, variants, gpus, args)
    manifest = {
        "args": vars(args),
        "selected_cases": cases,
        "variant_definitions": VARIANTS,
        "required_control_variants": required_control_variants,
        "planned_jobs": len(jobs),
        "jobs": jobs,
        "method_gate_claimed": False,
        "stage7_full_validation_claimed": False,
    }
    write_json(OUT_ROOT / "j4_read_weak_context_run_manifest.json", manifest)
    write_csv(
        OUT_ROOT / "job_manifest.csv",
        [
            {k: v for k, v in job.items() if k not in {"cmd"}}
            for job in jobs
        ],
    )
    if args.no_run:
        write_json(OUT_ROOT / "summary.json", {"status": "planned_not_run", "planned_jobs": len(jobs), "gate_pass": False})
        return

    if args.summarize_only:
        completed = []
        for job in jobs:
            p = Path(job["out_dir"]) / "job_summary.json"
            completed.append(json.loads(p.read_text(encoding="utf-8")) if p.is_file() else job)
    else:
        completed = run_jobs(jobs, gpus)
    write_csv(
        OUT_ROOT / "job_results.csv",
        [
            {k: v for k, v in job.items() if k not in {"cmd"}}
            for job in completed
        ],
    )
    rows, errors = evaluate_jobs(completed)
    comparisons = build_case_comparisons(
        rows,
        candidate_variant=args.candidate_variant,
        random_control_variant=args.random_control_variant,
        semantic_control_variant=args.semantic_control_variant,
        control_variants=required_control_variants,
    )
    decision = aggregate_decision(
        rows,
        comparisons,
        errors,
        candidate_variant=args.candidate_variant,
        random_control_variant=args.random_control_variant,
        semantic_control_variant=args.semantic_control_variant,
        control_variants=required_control_variants,
    )
    write_csv(OUT_ROOT / "rows.csv", rows)
    write_csv(OUT_ROOT / "per_case_candidate_comparison.csv", comparisons)
    write_csv(OUT_ROOT / "evaluation_errors.csv", errors)
    write_json(OUT_ROOT / "summary.json", decision)
    write_json(OUT_ROOT / "j4_read_weak_context_gate_summary.json", decision)
    write_csv(
        OUT_ROOT / "gate_checks.csv",
        [
            {"gate": "bad_metric_passes_nonempty", "pass": bool(decision["bad_metric_passes"]), "value": ",".join(decision["bad_metric_passes"])},
            {"gate": "required_control_variants", "pass": bool(decision["controls"]), "value": ",".join(decision["controls"])},
            {"gate": "good_safety_pass", "pass": decision["good_safety_pass"], "value": decision["good_j_short_proxy_worsen_ratio"]},
            {"gate": "trace_fidelity_pass", "pass": decision["trace_fidelity_pass"], "value": decision["trace_median_attention_mass_delta"]},
            {"gate": "global_safety_proxy_pass", "pass": decision["global_safety_proxy_pass"], "value": decision["trace_beta_frame_effective_median"]},
            {"gate": "stable_anchor_proxy_pass", "pass": decision["stable_anchor_proxy_pass"], "value": decision["stable_anchor_preservation_proxy"]},
            {"gate": "j4_mechanism_gate_pass", "pass": decision["gate_pass"], "value": decision["gate_pass"]},
        ],
    )
    write_csv(OUT_ROOT / "visual_manifest.csv", [])
    write_json(OUT_ROOT / "failure_attribution.json", decision)
    failure = (
        "# Track J4 READ Weak-Context Skip Pilot\n\n"
        f"Gate pass: {decision['gate_pass']}.\n\n"
        f"Failure reason: {decision['failure_reason'] or 'none'}\n\n"
        "This short-window pilot is not full validation and does not promote a runtime method.\n"
    )
    (OUT_ROOT / "failure_report.md").write_text(failure, encoding="utf-8")
    next_text = (
        "# What Would Have To Be True To Pass\n\n"
        "The candidate must improve READ_LOCAL L1 or L2 by at least 5%, beat both same-mass random and "
        "semantic-rotation controls by at least 5%, keep good-control J_short proxy worsen <=2%, move "
        "attention mass in the expected direction, and provide stable-anchor preservation evidence. "
        "The current stable-anchor field is only a retained-mass proxy, so a true stable-anchor trace is "
        "needed before any promotion.\n"
    )
    (OUT_ROOT / "what_would_have_to_be_true_to_pass.md").write_text(next_text, encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
