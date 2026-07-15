#!/usr/bin/env python3
"""Generate v119TF LB-LOGICAL TR pilot configs and command manifest."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_carrier_aware_augmented"
OLD_ROOT = ROOT / "results/acl2_v119tf_semantic_addressable_geometry_carrier_routing_representation_repair"
RUN_ROOT_NAME = os.environ.get("ACL2_V119_LB_STAGE2_RUN_ROOT_NAME", "stage2_lblogical_tr_pilot").strip()
RUN_ROOT = RESULT_ROOT / (RUN_ROOT_NAME or "stage2_lblogical_tr_pilot")
CONFIG_ROOT = RUN_ROOT / "configs"
WORKSPACE = RUN_ROOT / "workspace"
RAW_ACTION = RUN_ROOT / "raw_action"
FI_TRACE = RUN_ROOT / "fi_trace"
SEM_V3 = OLD_ROOT / "stage1_semv3_sidecar/semv3_prefix_rows.parquet"
SEM_SUPPORT = RUN_ROOT / "semv3_frame_support.csv"
BENCH = ROOT / "third_party/lingbot-map/benchmark"
CHECKPOINT = ROOT / "third_party/lingbot-map/checkpoints/lingbot-map-long.pt"
RAW_DATA_ROOT = ROOT / "data/kitti/dataset"
CONDA = Path("/mnt/data/users/chengshun.wang/miniconda3/bin/conda")
ENV_NAME = "loger"
PYTHONPATH = f"{ROOT / 'third_party/lingbot-map'}:{ROOT / 'third_party/lingbot-map/benchmark'}"
DATASET_TAG = os.environ.get("ACL2_V119_LB_STAGE2_DATASET_TAG", "lblogical_tr_pilot").strip() or "lblogical_tr_pilot"
LABEL_TAG = os.environ.get("ACL2_V119_LB_STAGE2_LABEL_TAG", DATASET_TAG).strip() or DATASET_TAG
VARIANT_SET = os.environ.get("ACL2_V119_LB_STAGE2_VARIANT_SET", "pilot").strip().lower() or "pilot"
GPU_IDS = [
    gpu.strip()
    for gpu in os.environ.get("ACL2_V119_LB_STAGE2_GPU_IDS", "0,1,2,3,4,5").split(",")
    if gpu.strip()
]
SEQ_LENGTHS = {"00": 4541}
SCALE_FRAMES = 8
AUTO_KEYFRAME_THRESHOLD = 320
ROLE_WEIGHTS = {
    "stable_landmark": 1.0,
    "weak_context": 0.65,
    "vegetation_repetitive": 0.45,
    "boundary_lowpurity": 0.2,
    "dynamic": 0.15,
    "sky_lowobs": 0.0,
    "unknown_lowtrust": 0.0,
}


def variant_rows() -> list[dict[str, str]]:
    if VARIANT_SET == "clbm_minimech":
        return [
            {
                "variant": "cp0_default_no_policy",
                "role": "default_control",
                "policy": "",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp1_page_qk_control",
                "role": "page_level_control",
                "policy": "TR1_QK_TOPK",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp2_internal_qk_topk4",
                "role": "internal_only_control",
                "policy": "CLBM_METRIC_QK_TOPK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "internal_qk_metric_control",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp3_metric_qualified_entry_retain",
                "role": "form_metric_qualified_entry_protect_retain_candidate",
                "policy": "CLBM_METRIC_QUALIFIED_RETAIN",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.03",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "metric_qualified_entry_protect_retain",
                "min_entry_age": "16",
                "max_entry_age": "",
                "retention_budget": "96",
                "local_lane_mode": "",
            },
            {
                "variant": "cp4_metric_qualified_read_boost",
                "role": "form_metric_qualified_read_boost_candidate",
                "policy": "CLBM_METRIC_QUALIFIED_READ_BOOST",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive,weak_context",
                "min_semantic_score": "0.02",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "metric_qualified_read_boost",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp5_metric_unqualified_long_memory_block",
                "role": "form_metric_unqualified_long_memory_block_candidate",
                "policy": "CLBM_UNQUALIFIED_LONG_MEMORY_BLOCK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "0.01",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "metric_unqualified_long_memory_block",
                "min_entry_age": "32",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp7_road_ground_metric_qualification_control",
                "role": "road_ground_metric_qualification_control",
                "policy": "CLBM_ROAD_GROUND_QUALIFICATION",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "road_ground_metric_qualification",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp8_low_baseline_repeated_view_control",
                "role": "low_baseline_repeated_view_control",
                "policy": "CLBM_LOW_BASELINE_REPEATED_VIEW",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "low_baseline_repeated_view_qualification",
                "min_entry_age": "",
                "max_entry_age": "16",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp9_dynamic_local_aligned_control",
                "role": "dynamic_but_locally_aligned_control",
                "policy": "CLBM_DYNAMIC_LOCAL_ALIGNED",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "dynamic,weak_context",
                "min_semantic_score": "0.01",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "dynamic_local_aligned_qualification",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp10_random_metric_control",
                "role": "same_count_random_metric_control",
                "policy": "CLBM_RANDOM_TOPK",
                "topk_entries": "4",
                "seed": "11952",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-M",
                "admission_mode": "metric_random_topk_control",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
        ]
    if VARIANT_SET == "clba_minimech":
        return [
            {
                "variant": "cp0_default_no_policy",
                "role": "default_control",
                "policy": "",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp1_page_qk_control",
                "role": "page_level_control",
                "policy": "TR1_QK_TOPK",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp2_internal_qk_topk4",
                "role": "internal_only_control",
                "policy": "CLBA_ANCHOR_LANDMARK_QK_TOPK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "internal_qk_anchor_control",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp3_anchor_landmark_qk",
                "role": "form_anchor_landmark_qk_candidate",
                "policy": "CLBA_ANCHOR_LANDMARK_QK_TOPK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.03",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "anchor_landmark_qk",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp4_persistent_landmark_only",
                "role": "form_persistent_landmark_only_candidate",
                "policy": "CLBA_PERSISTENT_LANDMARK_ONLY",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.03",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "persistent_landmark_only",
                "min_entry_age": "32",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp5_semantic_addressability_routing",
                "role": "form_semantic_addressability_routing_candidate",
                "policy": "CLBA_SEMANTIC_ADDRESSABILITY_ROUTING",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.01",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "semantic_addressability_routing",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp7_provenance_shuffle_control",
                "role": "same_class_track_shuffle_control",
                "policy": "CLBA_SEMANTIC_ADDRESSABILITY_ROUTING",
                "topk_entries": "4",
                "seed": "11941",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.01",
                "semantic_control": "provenance_shuffle",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "semantic_addressability_routing",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp8_role_swap_control",
                "role": "memory_role_swap_control",
                "policy": "CLBA_SEMANTIC_ADDRESSABILITY_ROUTING",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.01",
                "semantic_control": "role_swap",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "semantic_addressability_routing",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp9_reverse_control",
                "role": "reverse_anchor_control",
                "policy": "CLBA_REVERSE_QK_TOPK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.03",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "anchor_landmark_qk",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp10_random_control",
                "role": "same_token_count_random_control",
                "policy": "CLBA_RANDOM_TOPK",
                "topk_entries": "4",
                "seed": "11942",
                "role_filter": "stable_landmark,vegetation_repetitive",
                "min_semantic_score": "0.03",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-A",
                "admission_mode": "anchor_landmark_qk",
                "min_entry_age": "8",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
        ]
    if VARIANT_SET == "clbl_minimech":
        return [
            {
                "variant": "cp0_default_no_policy",
                "role": "default_control",
                "policy": "",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp1_page_qk_control",
                "role": "page_level_control",
                "policy": "TR1_QK_TOPK",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
                "local_lane_mode": "",
            },
            {
                "variant": "cp2_internal_qk_topk4",
                "role": "internal_only_control",
                "policy": "CLBL_LOGICAL_QK_TOPK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logical_gather",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp3_selected_query_logical_gather",
                "role": "form_selected_query_logical_gather_candidate",
                "policy": "CLBL_SELECTED_QUERY_LOGICAL_GATHER",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logical_gather",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp4_selected_query_logit_routing",
                "role": "form_selected_query_logit_routing_candidate",
                "policy": "CLBL_SELECTED_QUERY_LOGIT_ROUTING",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logit_routing",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp5_lane_tagged_value_routing",
                "role": "form_lane_tagged_value_routing_candidate",
                "policy": "CLBL_LANE_TAGGED_VALUE_ROUTING",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "lane_tagged_value_routing",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp7_provenance_shuffle_control",
                "role": "same_class_track_shuffle_control",
                "policy": "CLBL_SELECTED_QUERY_LOGIT_ROUTING",
                "topk_entries": "4",
                "seed": "11931",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "provenance_shuffle",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logit_routing",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp8_role_swap_control",
                "role": "memory_role_swap_control",
                "policy": "CLBL_SELECTED_QUERY_LOGIT_ROUTING",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "role_swap",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logit_routing",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp9_reverse_control",
                "role": "reverse_local_lane_control",
                "policy": "CLBL_REVERSE_QK_TOPK",
                "topk_entries": "4",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logical_gather",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
            {
                "variant": "cp10_random_control",
                "role": "same_token_count_random_control",
                "policy": "CLBL_RANDOM_TOPK",
                "topk_entries": "4",
                "seed": "11932",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-L",
                "admission_mode": "selected_query_logical_gather",
                "min_entry_age": "",
                "max_entry_age": "64",
                "retention_budget": "",
                "local_lane_mode": "all",
            },
        ]
    if VARIANT_SET == "clbp_minimech":
        return [
            {
                "variant": "cp0_default_no_policy",
                "role": "default_control",
                "policy": "",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp1_page_qk_control",
                "role": "page_level_control",
                "policy": "TR1_QK_TOPK",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "",
                "admission_mode": "",
                "min_entry_age": "",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp2_internal_qk_topk2",
                "role": "internal_only_control",
                "policy": "CLBP_LOGICAL_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "query_dependent_logical_retrieval",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp3_hard_persistent_qk",
                "role": "form_hard_logical_admission_candidate",
                "policy": "CLBP_HARD_LOGICAL_ADMISSION",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "hard_logical_admission",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp4_soft_tokentype_qk",
                "role": "form_soft_token_type_admission_candidate",
                "policy": "CLBP_SOFT_TOKEN_TYPE_ADMISSION",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "0.03",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "soft_token_type_admission",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp5_budgeted_retention_qk",
                "role": "form_budgeted_logical_retention_candidate",
                "policy": "CLBP_BUDGETED_LOGICAL_RETENTION",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.05",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "budgeted_logical_retention",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "128",
            },
            {
                "variant": "cp6_query_dependent_qk",
                "role": "form_query_dependent_logical_retrieval_candidate",
                "policy": "CLBP_QUERY_DEPENDENT_LOGICAL_RETRIEVAL",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "query_dependent_logical_retrieval",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp7_provenance_shuffle_control",
                "role": "same_class_track_shuffle_control",
                "policy": "CLBP_QUERY_DEPENDENT_LOGICAL_RETRIEVAL",
                "topk_entries": "2",
                "seed": "11921",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "provenance_shuffle",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "query_dependent_logical_retrieval",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp8_role_swap_control",
                "role": "memory_role_swap_control",
                "policy": "CLBP_QUERY_DEPENDENT_LOGICAL_RETRIEVAL",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "role_swap",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "query_dependent_logical_retrieval",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp9_reverse_control",
                "role": "reverse_persistent_control",
                "policy": "CLBP_REVERSE_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "query_dependent_logical_retrieval",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
            {
                "variant": "cp10_random_control",
                "role": "same_token_count_random_control",
                "policy": "CLBP_RANDOM_TOPK",
                "topk_entries": "2",
                "seed": "11922",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
                "explicit_carrier_branch": "C-LB-P",
                "admission_mode": "query_dependent_logical_retrieval",
                "min_entry_age": "64",
                "max_entry_age": "",
                "retention_budget": "",
            },
        ]
    if VARIANT_SET == "stable_role_semantic_controls":
        return [
            {
                "variant": "tr9_logical_provenance_shuffle_qk_topk2",
                "role": "fixed_internal_provenance_shuffle_control",
                "policy": "TR_LOGICAL_QK_TOPK",
                "topk_entries": "2",
                "seed": "11901",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "provenance_shuffle",
            },
            {
                "variant": "tr10_logical_role_swap_qk_topk2",
                "role": "fixed_carrier_memory_role_swap_control",
                "policy": "TR_LOGICAL_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "role_swap",
            },
        ]
    if VARIANT_SET == "stable_role_ablation":
        return [
            {
                "variant": "tr0_default_no_policy",
                "role": "default_control",
                "policy": "",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
            },
            {
                "variant": "tr1_page_qk_topk",
                "role": "page_level_control",
                "policy": "TR1_QK_TOPK",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
            },
            {
                "variant": "tr8_logical_internal_qk_topk2",
                "role": "internal_only_ablation",
                "policy": "TR_LOGICAL_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
            },
            {
                "variant": "tr5_logical_stable_qk_topk",
                "role": "candidate_logical_stable_qk",
                "policy": "TR_LOGICAL_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
            },
            {
                "variant": "tr6_logical_stable_reverse_qk",
                "role": "stable_reverse_control",
                "policy": "TR_LOGICAL_REVERSE_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
            },
            {
                "variant": "tr7_logical_stable_random_seed00",
                "role": "stable_matched_random_control",
                "policy": "TR_LOGICAL_RANDOM_TOPK",
                "topk_entries": "2",
                "seed": "11900",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
            },
        ]
    if VARIANT_SET == "stable_role_repair":
        return [
            {
                "variant": "tr0_default_no_policy",
                "role": "default_control",
                "policy": "",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
            },
            {
                "variant": "tr1_page_qk_topk",
                "role": "page_level_control",
                "policy": "TR1_QK_TOPK",
                "topk_entries": "",
                "seed": "",
                "role_filter": "",
                "min_semantic_score": "",
                "semantic_control": "",
            },
            {
                "variant": "tr5_logical_stable_qk_topk",
                "role": "candidate_logical_stable_qk",
                "policy": "TR_LOGICAL_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
            },
            {
                "variant": "tr6_logical_stable_reverse_qk",
                "role": "stable_reverse_control",
                "policy": "TR_LOGICAL_REVERSE_QK_TOPK",
                "topk_entries": "2",
                "seed": "",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
            },
            {
                "variant": "tr7_logical_stable_random_seed00",
                "role": "stable_matched_random_control",
                "policy": "TR_LOGICAL_RANDOM_TOPK",
                "topk_entries": "2",
                "seed": "11900",
                "role_filter": "stable_landmark",
                "min_semantic_score": "0.09",
                "semantic_control": "",
            },
        ]
    return [
        {
            "variant": "tr0_default_no_policy",
            "role": "default_control",
            "policy": "",
            "topk_entries": "",
            "seed": "",
            "role_filter": "",
            "min_semantic_score": "",
            "semantic_control": "",
        },
        {
            "variant": "tr1_page_qk_topk",
            "role": "page_level_control",
            "policy": "TR1_QK_TOPK",
            "topk_entries": "",
            "seed": "",
            "role_filter": "",
            "min_semantic_score": "",
            "semantic_control": "",
        },
        {
            "variant": "tr2_logical_qk_topk",
            "role": "candidate_logical_qk",
            "policy": "TR_LOGICAL_QK_TOPK",
            "topk_entries": "4",
            "seed": "",
            "role_filter": "",
            "min_semantic_score": "",
            "semantic_control": "",
        },
        {
            "variant": "tr3_logical_reverse_qk",
            "role": "reverse_control",
            "policy": "TR_LOGICAL_REVERSE_QK_TOPK",
            "topk_entries": "4",
            "seed": "",
            "role_filter": "",
            "min_semantic_score": "",
            "semantic_control": "",
        },
        {
            "variant": "tr4_logical_random_seed00",
            "role": "matched_random_control",
            "policy": "TR_LOGICAL_RANDOM_TOPK",
            "topk_entries": "4",
            "seed": "11900",
            "role_filter": "",
            "min_semantic_score": "",
            "semantic_control": "",
        },
    ]


VARIANTS = variant_rows()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaml_list(items: list[int | str]) -> str:
    return "[" + ", ".join(f'"{item}"' if isinstance(item, str) else str(int(item)) for item in items) + "]"


def default_frozen_indices(num_frames: int) -> list[int]:
    interval = math.ceil(num_frames / AUTO_KEYFRAME_THRESHOLD)
    stream = [idx for idx in range(num_frames) if idx >= SCALE_FRAMES]
    return [idx for pos, idx in enumerate(stream) if interval <= 1 or pos % interval == 0]


def semantic_score(row: Any) -> float:
    role = str(row.dominant_role_prefix)
    return float(
        ROLE_WEIGHTS.get(role, 0.0)
        * float(row.semv3_role_prior_prefix)
        * float(row.semv3_shape_score_prefix)
        * float(row.semv3_reobs_score_prefix)
        * min(1.0, 4.0 * float(row.semv3_visibility_prefix))
    )


def build_semantic_support() -> list[dict[str, Any]]:
    if not SEM_V3.exists():
        raise FileNotFoundError(SEM_V3)
    df = pd.read_parquet(
        SEM_V3,
        columns=[
            "seq",
            "frame_id",
            "track_id",
            "dominant_role_prefix",
            "dominant_label_prefix",
            "semv3_visibility_prefix",
            "semv3_role_prior_prefix",
            "semv3_shape_score_prefix",
            "semv3_reobs_score_prefix",
            "semv3_identity_key",
        ],
    )
    rows: list[dict[str, Any]] = []
    for (seq, frame_id), frame_df in df.groupby([df["seq"].astype(str), df["frame_id"].astype(int)], sort=True):
        best_score = -1.0
        best_row = None
        for row in frame_df.itertuples(index=False):
            score = semantic_score(row)
            if score > best_score:
                best_score = score
                best_row = row
        rows.append(
            {
                "schema": "acl2_v119tf_semv3_frame_support_for_lblogical_tr_v1",
                "seq": str(seq),
                "frame_id": int(frame_id),
                "max_semantic_persistence_prefix": float(max(best_score, 0.0)),
                "unique_track_count": int(len(frame_df)),
                "best_track_id_by_semantic_persistence": int(getattr(best_row, "track_id", -1)),
                "best_track_id": int(getattr(best_row, "track_id", -1)),
                "best_track_role": str(getattr(best_row, "dominant_role_prefix", "")),
                "best_track_label": str(getattr(best_row, "dominant_label_prefix", "")),
                "best_identity_key": str(getattr(best_row, "semv3_identity_key", "")),
                "score_formula": "role_weight*role_prior*shape_score*reobs_score*min(1,4*visibility)",
            }
        )
    write_csv(SEM_SUPPORT, rows)
    return rows


def method_yaml(method: str, frozen: list[int]) -> str:
    return f"""model: lingbot_map
env: {ENV_NAME}
_checkpoint: {CHECKPOINT}
_device: cuda
_use_amp: true
_use_sdpa: false
_image_size: 518
_patch_size: 14
_enable_3d_rope: true
_num_scale_frames: 8
_max_frame_num: 1024
_kv_cache_sliding_window: 64
_kv_cache_scale_frames: 8
_auto_keyframe_threshold: 320
_area_budget: 255000
_align: 14
_mode: streaming
_keyframe_interval: auto
_keyframe_schedule_mode: global_frozen
_frozen_keyframe_indices: {yaml_list(frozen)}
_stage4_action_mode: force_non_keyframe
_stage4_action_label: v119_{LABEL_TAG}_{method}
"""


def base_yaml(workspace: Path, dataset: str, method: str) -> str:
    return f"""workspace: {workspace}

evaluation:
  traj:
    enable: true
    vis: true
  auc:
    enable: false
  depth:
    enable: false
  points:
    enable: false

datasets:
  - {dataset}

methods:
  - {method}
"""


def main() -> None:
    support_rows = build_semantic_support()
    logs = RUN_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    RAW_ACTION.mkdir(parents=True, exist_ok=True)
    FI_TRACE.mkdir(parents=True, exist_ok=True)
    env_prefix = f"PATH={CONDA.parent}:$PATH PYTHONPATH={PYTHONPATH}"
    run_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []

    for seq, num_frames in SEQ_LENGTHS.items():
        dataset = f"kitti_v119_{DATASET_TAG}_seq{seq}"
        dataset_cfg = CONFIG_ROOT / "datasets" / f"{dataset}.yaml"
        write_text(
            dataset_cfg,
            f"""dataset: kitti
raw_data_root: {RAW_DATA_ROOT}
_target_size: [504, 280]
_sequences: {yaml_list([seq])}
""",
        )
        frozen = default_frozen_indices(num_frames)
        prepare_method = f"lingbot_map_v119_{DATASET_TAG}_{VARIANTS[0]['variant']}_{seq}"
        prepare_cfg = CONFIG_ROOT / f"kitti_{DATASET_TAG}_prepare_seq{seq}.yaml"
        write_text(prepare_cfg, base_yaml(WORKSPACE, dataset, prepare_method))
        prepare_log = logs / f"prepare_seq{seq}.log"
        run_rows.append(
            {
                "phase": "prepare",
                "seq": seq,
                "variant": "prepare",
                "gpu": "",
                "cwd": str(BENCH),
                "config": str(prepare_cfg),
                "dataset": dataset,
                "method": "",
                "log": str(prepare_log),
                "command": (
                    f"{env_prefix} {CONDA} run -n {ENV_NAME} --no-capture-output "
                    f"python prepare.py --config {prepare_cfg} --force > {prepare_log} 2>&1"
                ),
            }
        )
        for idx, variant in enumerate(VARIANTS):
            method = f"lingbot_map_v119_{DATASET_TAG}_{variant['variant']}_{seq}"
            method_cfg = CONFIG_ROOT / "methods" / f"{method}.yaml"
            base_cfg = CONFIG_ROOT / f"kitti_{DATASET_TAG}_{variant['variant']}_seq{seq}.yaml"
            write_text(method_cfg, method_yaml(method, frozen))
            write_text(base_cfg, base_yaml(WORKSPACE, dataset, method))
            action_file = RAW_ACTION / f"{dataset}_{seq}_{method}.jsonl"
            trace_file = FI_TRACE / f"{dataset}_{seq}_{method}.jsonl"
            action_file.write_text("", encoding="utf-8")
            trace_file.write_text("", encoding="utf-8")
            if not GPU_IDS:
                raise ValueError("ACL2_V119_LB_STAGE2_GPU_IDS produced no usable GPU ids")
            gpu = GPU_IDS[idx % len(GPU_IDS)]
            policy_env = ""
            if variant["policy"]:
                policy_env += f" ACL2_V118_LB_STAGE4_POLICY={variant['policy']}"
            if variant["topk_entries"]:
                policy_env += f" ACL2_V119_LB_LOGICAL_TOPK_ENTRIES={variant['topk_entries']}"
            if variant["seed"]:
                policy_env += f" ACL2_V118_LB_STAGE4_RANDOM_SEED={variant['seed']}"
            if variant.get("role_filter"):
                policy_env += f" ACL2_V119_LB_LOGICAL_ROLE_FILTER={variant['role_filter']}"
            if variant.get("min_semantic_score"):
                policy_env += f" ACL2_V119_LB_LOGICAL_MIN_SEMANTIC_SCORE={variant['min_semantic_score']}"
            if variant.get("semantic_control"):
                policy_env += f" ACL2_V119_LB_LOGICAL_SEMANTIC_CONTROL={variant['semantic_control']}"
            if variant.get("explicit_carrier_branch"):
                policy_env += f" ACL2_V119_LB_EXPLICIT_CARRIER_BRANCH={variant['explicit_carrier_branch']}"
            if variant.get("admission_mode"):
                policy_env += f" ACL2_V119_LB_LOGICAL_ADMISSION_MODE={variant['admission_mode']}"
            if variant.get("min_entry_age"):
                policy_env += f" ACL2_V119_LB_LOGICAL_MIN_ENTRY_AGE={variant['min_entry_age']}"
            if variant.get("max_entry_age"):
                policy_env += f" ACL2_V119_LB_LOGICAL_MAX_ENTRY_AGE={variant['max_entry_age']}"
            if variant.get("retention_budget"):
                policy_env += f" ACL2_V119_LB_LOGICAL_RETENTION_BUDGET={variant['retention_budget']}"
            if variant.get("local_lane_mode"):
                policy_env += f" ACL2_V119_LB_LOCAL_LANE_MODE={variant['local_lane_mode']}"
            common_env = (
                f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} "
                f"ACL2_V105_STAGE4_ACTION_FILE={action_file} "
                f"ACL2_V105_STAGE4_ACTION_LABEL=v119_{LABEL_TAG}_{variant['variant']} "
                f"ACL2_V105_GCA_TRACE_DATASET={dataset} "
                f"ACL2_V105_GCA_TRACE_SEQ={seq} "
                f"ACL2_V105_GCA_TRACE_METHOD={method} "
                f"ACL2_V118_LB_FI_PROVENANCE_FILE={trace_file} "
                f"ACL2_V118_LB_PROVENANCE_SEQ={seq} "
                f"ACL2_V118_LB_SEMANTIC_FRAME_SUPPORT_FILE={SEM_SUPPORT} "
                f"ACL2_V118_LB_SEMANTIC_FRAME_SCORE_FIELD=max_semantic_persistence_prefix "
                f"ACL2_V118_LB_FI_PROVENANCE_MAX_ROWS=800000"
                f"{policy_env}"
            )
            run_log = logs / f"run_{variant['variant']}_seq{seq}_gpu{gpu}.log"
            eval_log = logs / f"evaluate_{variant['variant']}_seq{seq}.log"
            run_rows.append(
                {
                    "phase": "run_worker",
                    "seq": seq,
                    "variant": variant["variant"],
                    "role": variant["role"],
                    "policy": variant["policy"],
                    "role_filter": variant.get("role_filter", ""),
                    "min_semantic_score": variant.get("min_semantic_score", ""),
                    "semantic_control": variant.get("semantic_control", ""),
                    "explicit_carrier_branch": variant.get("explicit_carrier_branch", ""),
                    "admission_mode": variant.get("admission_mode", ""),
                    "min_entry_age": variant.get("min_entry_age", ""),
                    "max_entry_age": variant.get("max_entry_age", ""),
                    "retention_budget": variant.get("retention_budget", ""),
                    "local_lane_mode": variant.get("local_lane_mode", ""),
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(base_cfg),
                    "dataset": dataset,
                    "method": method,
                    "action_file": str(action_file),
                    "fi_trace": str(trace_file),
                    "log": str(run_log),
                    "command": (
                        f"{common_env} {CONDA} run -n {ENV_NAME} --no-capture-output "
                        f"python run_worker.py --config {base_cfg} --method {method} "
                        f"--dataset {dataset} --scene {seq} --force > {run_log} 2>&1"
                    ),
                }
            )
            run_rows.append(
                {
                    "phase": "evaluate",
                    "seq": seq,
                    "variant": variant["variant"],
                    "role": variant["role"],
                    "policy": variant["policy"],
                    "role_filter": variant.get("role_filter", ""),
                    "min_semantic_score": variant.get("min_semantic_score", ""),
                    "semantic_control": variant.get("semantic_control", ""),
                    "explicit_carrier_branch": variant.get("explicit_carrier_branch", ""),
                    "admission_mode": variant.get("admission_mode", ""),
                    "min_entry_age": variant.get("min_entry_age", ""),
                    "max_entry_age": variant.get("max_entry_age", ""),
                    "retention_budget": variant.get("retention_budget", ""),
                    "local_lane_mode": variant.get("local_lane_mode", ""),
                    "gpu": gpu,
                    "cwd": str(BENCH),
                    "config": str(base_cfg),
                    "dataset": dataset,
                    "method": method,
                    "action_file": "",
                    "fi_trace": "",
                    "log": str(eval_log),
                    "command": (
                        f"{env_prefix} CUDA_VISIBLE_DEVICES={gpu} {CONDA} run -n {ENV_NAME} --no-capture-output "
                        f"python evaluate.py --config {base_cfg} --force > {eval_log} 2>&1"
                    ),
                }
            )
            config_rows.append(
                {
                    "schema": "acl2_v119tf_lblogical_tr_pilot_config_row_v1",
                    "seq": seq,
                    "variant": variant["variant"],
                    "role": variant["role"],
                    "policy": variant["policy"],
                    "role_filter": variant.get("role_filter", ""),
                    "min_semantic_score": variant.get("min_semantic_score", ""),
                    "semantic_control": variant.get("semantic_control", ""),
                    "explicit_carrier_branch": variant.get("explicit_carrier_branch", ""),
                    "admission_mode": variant.get("admission_mode", ""),
                    "min_entry_age": variant.get("min_entry_age", ""),
                    "max_entry_age": variant.get("max_entry_age", ""),
                    "retention_budget": variant.get("retention_budget", ""),
                    "local_lane_mode": variant.get("local_lane_mode", ""),
                    "method": method,
                    "dataset": dataset,
                    "method_cfg": rel(method_cfg),
                    "base_cfg": rel(base_cfg),
                    "action_file": rel(action_file),
                    "fi_trace": rel(trace_file),
                    "gpu": gpu,
                    "frozen_keyframe_count": len(frozen),
                }
            )

    write_csv(RUN_ROOT / "run_manifest.csv", run_rows)
    write_csv(RUN_ROOT / "config_rows.csv", config_rows)
    summary = {
        "schema": "acl2_v119tf_lblogical_tr_pilot_config_summary_v1",
        "run_root": rel(RUN_ROOT),
        "dataset_tag": DATASET_TAG,
        "label_tag": LABEL_TAG,
        "variant_set": VARIANT_SET,
        "semv3_source": rel(SEM_V3),
        "semantic_support_csv": rel(SEM_SUPPORT),
        "semantic_support_row_count": len(support_rows),
        "variant_count": len(VARIANTS),
        "variants": [row["variant"] for row in VARIANTS],
        "gpu_ids": GPU_IDS,
        "truthfulness_boundary": "pilot config generation only; no run/eval metrics are claimed here",
        "run_manifest": rel(RUN_ROOT / "run_manifest.csv"),
        "config_rows": rel(RUN_ROOT / "config_rows.csv"),
    }
    write_text(RUN_ROOT / "config_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
