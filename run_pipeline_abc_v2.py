#!/usr/bin/env python3
"""
LoGeR Hybrid Memory Pipeline v2.

This is the pipeline2 implementation entry point described in
``docs/pipeline2``.  The important runtime difference from
``run_pipeline_abc.py`` is the two-pass chunk protocol:

  Pass 1: probe LoGeR from committed hybrid state, collect geometry/caches.
  Stages B/C/D: build memory-control priors from probe outputs.
  Pass 2: controlled LoGeR from the same committed state, then commit the
          controlled hybrid state for the next chunk.

The first executable version implements the v2 interface and the TTT-update
subpath plus real identity read-path hook sites for frame attention, SWA read,
TTT apply, and chunk attention.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from loger.pipeline.dynamic_cue_extractor import DynamicCueExtractor, CueOutput
from loger.pipeline.geometry_backbone import GeometryOutput, LoGeRGeometryBackbone, load_images as loger_load_images
from loger.pipeline.hybrid_memory_controller import (
    HybridMemoryController,
    HybridMemoryControlPrior,
    HybridMemoryResult,
    HybridMemoryState,
    ProbeOutput,
    hybrid_state_fingerprint,
)
from loger.pipeline.semantic_prior_generator import PriorOutput, SemanticPriorGenerator
from loger.pipeline.video_masklet_frontend import MaskletOutput
from loger.utils.rotation import mat_to_quat

from inference_dynamic_cue_extractor import print_cue_output
from run_geometry_backbone_inference import collect_image_paths as collect_image_paths_geo
from run_geometry_backbone_inference import print_geometry_output

# Reuse v1's stable utility surface.  The v2 orchestration below is separate.
from run_pipeline_abc import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    DEFAULT_SAM31_CHECKPOINT,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_VIDEO_STUFF_PROMPTS,
    DEFAULT_VIDEO_THING_PROMPTS,
    _apply_debug_prior_mode,
    _apply_prior_policy,
    _append_prior_debug_jsonl,
    _empty_masklet_output,
    _merge_external_window_predictions,
    _move_tensor_tree_to_device,
    _offload_backbone_to_cpu,
    _offload_stage_c_frontend_to_cpu,
    _rebuild_batched_raw_window,
    _run_stage_c_lazy,
    _ensure_backbone_on_device,
    align_chunk_geometry_outputs,
    build_timestamps_for_output,
    create_video,
    load_images_tensor,
    merge_chunk_tensor_tail_trim,
    print_masklet_output,
    print_prior_output,
    print_write_result,
    score_masklets_by_cdyn,
    split_into_chunks,
    write_trajectory_txt,
)


HYBRID_MEMORY_MODES = (
    "native",
    "unity_replay",
    "identity_hooks",
    "ttt_write_only",
    "read_path_only",
    "hybrid",
    "probe_only",
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LoGeR Hybrid Memory Pipeline v2 (two-pass probe/control).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True, help="Image folder or video file.")
    p.add_argument("--output_video", default="results/pipeline_v2.mp4")
    p.add_argument("--output_pt", default=None, help="Save merged outputs as .pt file.")
    p.add_argument("--output_txt", default=None, help="Save merged camera trajectory in TUM format.")
    p.add_argument("--save_frames", default=None)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--geometry_eval_mode", action="store_true",
                   help="Geometry-only mode: skip B/C/D; use native/unity/probe modes.")

    # -- Chunk scheduling -------------------------------------------------
    p.add_argument("--chunk_size", type=int, default=0, help="Frames per chunk (0 = all frames).")
    p.add_argument("--chunk_overlap", type=int, default=2)

    # -- Stage A ----------------------------------------------------------
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap_size", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=0)
    p.add_argument(
        "--ttt_freeze_chunks",
        default="",
        help="Comma-separated chunk indices whose TTT write commit is discarded while other memory state is kept. Diagnostic only.",
    )
    p.add_argument(
        "--ttt_semantic_write_scale_chunks",
        default="",
        help=(
            "Comma-separated CHUNK:SCALE entries. For probe_ttt_write commits, "
            "preserve native LoGeR TTT and scale only the semantic write delta "
            "against the native provisional TTT state. Diagnostic only."
        ),
    )
    p.add_argument("--se3", action="store_true", default=None)
    p.add_argument("--geometry_edge_rtol", type=float, default=0.03)
    p.add_argument("--stage_memory_swap", type=int, default=1)

    # -- Stage B ----------------------------------------------------------
    p.add_argument("--k_intra", type=int, default=3)
    p.add_argument("--disable_attention_prior", action="store_true")
    p.add_argument("--support_time_decay", type=float, default=2.0)
    p.add_argument("--support_temporal_weight", type=float, default=0.35)
    p.add_argument("--support_affinity_weight", type=float, default=0.45)
    p.add_argument("--support_static_weight", type=float, default=0.20)
    p.add_argument("--sigma_pt", type=float, default=0.25)
    p.add_argument("--stageb_proxy_mode", choices=["same_pixel", "reprojection"], default="same_pixel")
    p.add_argument("--stageb_nonocc_dynamic", type=int, default=0)
    p.add_argument("--tau_occ", type=float, default=0.05)
    p.add_argument("--alpha_1", type=float, default=0.8)
    p.add_argument("--alpha_3", type=float, default=0.5)
    p.add_argument("--attn_stat_fusion_weight", type=float, default=0.35)
    p.add_argument("--dyn_fusion_mode",
                   choices=["explicit", "implicit", "max", "soft_or", "avg", "addclip", "calibrated_soft_or"],
                   default="max")
    p.add_argument("--implicit_weight", type=float, default=1.0)
    p.add_argument("--implicit_gate_floor", type=float, default=0.25)
    p.add_argument("--implicit_calib_min_range", type=float, default=1e-3)
    p.add_argument("--lambda_s", type=float, default=1.0)
    p.add_argument("--lambda_a", type=float, default=0.5)
    p.add_argument("--lambda_d", type=float, default=0.8)
    p.add_argument("--lambda_o", type=float, default=0.3)
    p.add_argument("--lambda_u", type=float, default=0.5)

    # -- Stage C ----------------------------------------------------------
    p.add_argument("--stage_c_mode", choices=["reference", "none"], default="reference")
    p.add_argument("--sam_backend", choices=["sam2", "sam3", "sam31_multiplex"], default="sam31_multiplex")
    p.add_argument("--tracker_backend", default="sam2",
                   choices=["sam2", "edgetam", "cutie", "efficientsam3", "efficient"],
                   help="Compatibility arg; VideoMaskletFrontend ignores it.")
    p.add_argument("--sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    p.add_argument("--sam2_model_cfg", default=DEFAULT_SAM2_MODEL_CFG)
    p.add_argument("--sam3_checkpoint", default=DEFAULT_SAM3_CHECKPOINT)
    p.add_argument("--sam31_checkpoint", default=DEFAULT_SAM31_CHECKPOINT)
    p.add_argument("--sam31_offload_video_to_cpu", type=int, default=1)
    p.add_argument("--sam31_offload_outputs_to_cpu", type=int, default=1)
    p.add_argument("--sam31_offload_sam_during_detection", type=int, default=0)
    p.add_argument("--sam31_text_track_labels", default="person")
    p.add_argument("--sam31_direct_text_prompt_labels", default="person")
    p.add_argument("--sam31_direct_text_prompt_frame_count", type=int, default=1)
    p.add_argument("--sam31_structure_prompt_labels", default="wall,floor,ceiling")
    p.add_argument("--sam31_structure_prompt_frame_count", type=int, default=1)
    p.add_argument("--sam31_structure_prompt_chunk_stride", type=int, default=1)
    p.add_argument("--sam31_person_refresh_prompt_frames", type=int, default=1)
    p.add_argument("--sam31_nontext_object_prompt_budget", type=int, default=0)
    p.add_argument("--sam31_nontext_object_prompt_min_support", type=int, default=2)
    p.add_argument("--sam31_text_object_prompt_budget", type=int, default=0)
    p.add_argument("--sam31_nontext_sparse_support", type=int, default=1)
    p.add_argument("--sam31_max_text_prompt_objects", type=int, default=0)
    p.add_argument("--sam31_max_internal_objects", type=int, default=16)
    p.add_argument("--sam31_max_movable_objects", type=int, default=8)
    p.add_argument("--sam31_max_static_objects", type=int, default=2)
    p.add_argument("--sam31_max_structure_objects", type=int, default=3)
    p.add_argument("--detector", choices=["gdino", "yoloe"], default="yoloe")
    p.add_argument("--gdino_config", default=None)
    p.add_argument("--gdino_checkpoint", default=None)
    p.add_argument("--yoloe_model", default="yoloe-11l-seg.pt")
    p.add_argument("--yoloe_batch_size", type=int, default=4)
    p.add_argument("--yoloe_imgsz", type=int, default=0)
    p.add_argument("--ann_frame_idx", type=int, default=0)
    p.add_argument("--discovery_frame_stride", type=int, default=4)
    p.add_argument("--max_thing_objects", type=int, default=24)
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--thing_prompts", default=DEFAULT_VIDEO_THING_PROMPTS)
    p.add_argument("--stuff_prompts", default=DEFAULT_VIDEO_STUFF_PROMPTS)

    # -- Stage D / Hybrid Memory Controller -------------------------------
    p.add_argument("--hybrid_memory_mode", choices=HYBRID_MEMORY_MODES, default=None,
                   help="v2 mode. If omitted, maps legacy --ttt_write_mode to v2.")
    p.add_argument("--hmc_commit_mode",
                   choices=("controlled", "probe_native", "split_ttt_native", "probe_ttt_write"),
                   default="controlled",
                   help="Phase C v5 commit isolation: choose which provisional memory becomes H_next.")
    p.add_argument("--ttt_write_mode", choices=["semantic", "unity_replay", "native"], default="semantic",
                   help="Legacy compatibility: semantic->ttt_write_only, unity_replay/native keep their meaning.")
    p.add_argument("--lambda_min", type=float, default=1.0)
    p.add_argument("--lambda_max", type=float, default=1.0)
    p.add_argument("--a_min_special", type=float, default=1.0)
    p.add_argument("--a_token_floor", type=float, default=0.0)
    p.add_argument("--prior_policy", choices=["suppressive", "eta_mean_preserving"], default="eta_mean_preserving")
    p.add_argument("--mp_alpha", type=float, default=0.1)
    p.add_argument("--mp_min", type=float, default=0.8)
    p.add_argument("--mp_max", type=float, default=1.2)
    p.add_argument("--mp_score_source",
                   choices=["e_patch", "a_patch", "anchor", "positive", "dyn", "unc", "occ", "anchor_minus_dyn"],
                   default="dyn")
    p.add_argument("--prior_branch_mask", default="0",
                   help="Default v2/BL01 setting controls only TTT branch 0.")
    p.add_argument("--prior_layer_mode", choices=["all", "early", "late", "middle", "single"], default="all")
    p.add_argument("--prior_single_layer", type=int, default=-1)
    p.add_argument("--prior_layer_branch_policy", default=None,
                   help="Optional inclusive layer-range branch overrides, e.g. '0-5:all;6-11:0;12-17:none'.")
    p.add_argument("--ttt_write_delta_scale", type=float, default=1.0,
                   help="Post-replay scale for the actual fast-weight delta W_new-W_old.")
    p.add_argument("--ttt_write_delta_scales", default=None,
                   help="Optional branch-wise post-replay delta scales as w0,w1,w2. Overrides the global scale per branch.")
    p.add_argument("--ttt_write_native_mix_scales", default=None,
                   help="Optional branch-wise semantic-vs-native replay mix as w0,w1,w2; 1 keeps semantic replay, 0 keeps native write.")
    p.add_argument("--ttt_write_prior_transform_mode",
                   choices=("none", "focal_static", "static_focal", "pow", "anti_dynamic", "dynamic_anti", "signed_center", "center_signed", "signed_focal", "focal_signed", "anti_dynamic_norm", "dynamic_anti_norm", "signed_center_norm", "center_signed_norm", "signed_focal_norm", "focal_signed_norm"),
                   default="none",
                   help="Optional transform applied to TTT write prior before it multiplies replay lr. Anti/signed modes allow low-prior tokens to push negative updates.")
    p.add_argument("--ttt_write_prior_anti_scale", type=float, default=0.0)
    p.add_argument("--ttt_write_prior_gamma", type=float, default=1.0)
    p.add_argument("--ttt_write_replay_feature_gate_mode",
                   choices=(
                       "none",
                       "k_centered", "key_centered",
                       "v_centered", "value_centered",
                       "kv_centered", "both_centered",
                       "k_frame_static_center", "key_frame_static_center",
                       "v_frame_static_center", "value_frame_static_center",
                       "kv_frame_static_center", "both_frame_static_center",
                       "frame_static_center",
                   ),
                   default="none",
                   help="Optionally center-gate cached TTT replay K/V residuals for low-prior tokens before zeropower.")
    p.add_argument("--ttt_write_replay_feature_gate_rho", type=float, default=0.0)
    p.add_argument("--ttt_write_replay_feature_gate_min", type=float, default=0.5)
    p.add_argument("--ttt_write_replay_feature_gate_branch_mask", default="all",
                   help="When replay feature gate is enabled, take only these final TTT branches from the gated replay; other branches keep native semantic replay.")
    p.add_argument("--ttt_write_replay_token_filter_mode",
                   choices=(
                       "none",
                       "static_topk", "dynamic_veto", "per_frame_static_topk",
                       "scoped_dynamic_veto", "overlap_dynamic_veto",
                       "scoped_static_topk", "overlap_static_topk",
                   ),
                   default="none",
                   help="Hard-filter TTT replay tokens before zeropower aggregation.")
    p.add_argument("--ttt_write_replay_token_filter_ratio", type=float, default=1.0)
    p.add_argument("--ttt_write_replay_token_filter_threshold", type=float, default=1.0)
    p.add_argument("--ttt_write_replay_token_filter_scope",
                   choices=("all", "tail_overlap", "head_overlap", "both_overlap"),
                   default="all",
                   help="Temporal scope for scoped replay token filters. Outside this scope, tokens are kept.")
    p.add_argument("--ttt_write_replay_token_filter_branch_mask", default="all",
                   help="When hard replay token filter is enabled, take filtered replay only for these TTT branches.")
    p.add_argument("--ttt_write_replay_token_filter_blend", type=float, default=1.0,
                   help="Blend full replay toward filtered replay for selected branches; 0 keeps full replay, 1 is hard filtering.")
    p.add_argument("--ttt_write_replay_token_filter_blend_mode",
                   choices=(
                       "linear",
                       "project_anti_dynamic", "proj_anti_dynamic",
                       "anti_dynamic_project", "project_dynamic_residual",
                       "aligned_dynamic", "align_dynamic", "aligned_dyn", "align_dyn",
                       "ttl_dynamic", "transient_dynamic", "dynamic_ttl",
                       "ttl_aligned_dynamic", "transient_aligned_dynamic",
                       "aligned_dynamic_ttl", "align_dynamic_ttl",
                   ),
                   default="linear",
                   help="How to combine full replay and filtered replay. ttl_* stores dynamic residuals for one-hop subtraction on the next TTT commit.")
    p.add_argument("--ttt_write_transient_delta_subtract_scale", type=float, default=0.0,
                   help="Subtract the previous chunk's stored transient dynamic delta from final TTT commit. 0 disables.")
    p.add_argument("--ttt_write_transient_delta_branch_mask", default="0",
                   help="Branches affected by transient dynamic delta subtraction.")
    p.add_argument("--ttt_write_commit_ema_alpha", type=float, default=1.0,
                   help="Final committed TTT fast-weight EMA alpha against W_m, after all replay/mix/gate steps.")
    p.add_argument("--ttt_write_commit_ema_branch_mask", default="all",
                   help="Apply commit EMA only to these TTT branches; other branches keep their candidate update.")
    p.add_argument("--ttt_write_native_delta_gate_mode",
                   choices=("none", "cosine", "cosine_soft", "cap", "cosine_cap"),
                   default="none",
                   help="Post-replay gate for semantic correction relative to native TTT replay continuity.")
    p.add_argument("--ttt_write_native_delta_gate_min_cos", type=float, default=0.0)
    p.add_argument("--ttt_write_native_delta_gate_fallback", type=float, default=0.0)
    p.add_argument("--ttt_write_native_delta_gate_cap_ratio", type=float, default=1.0)
    p.add_argument("--ttt_write_native_delta_gate_branch_mask", default="all",
                   help="Branches affected by post-replay native delta gate.")
    p.add_argument("--ttt_write_commit_filter_mode",
                   choices=("none", "native_to_candidate_by_risk", "old_decay_by_risk"),
                   default="none",
                   help="Post-replay commit-only TTT propagation filter; does not change current-chunk controlled output.")
    p.add_argument("--ttt_write_commit_filter_risk_source",
                   choices=("d_tok", "write_prior"),
                   default="d_tok",
                   help="Risk map for commit filter: raw dynamic D_tok, or 1-write_prior.")
    p.add_argument("--ttt_write_commit_filter_scope",
                   choices=("all", "tail_overlap", "head_overlap", "both_overlap"),
                   default="tail_overlap",
                   help="Temporal token scope used to estimate risk for TTT commit filtering.")
    p.add_argument("--ttt_write_commit_filter_stat",
                   choices=("mean", "q90", "q75", "max", "mass_gt_05"),
                   default="mean",
                   help="Statistic over the selected risk scope.")
    p.add_argument("--ttt_write_commit_filter_base", type=float, default=0.0)
    p.add_argument("--ttt_write_commit_filter_gain", type=float, default=1.0)
    p.add_argument("--ttt_write_commit_filter_min", type=float, default=0.0)
    p.add_argument("--ttt_write_commit_filter_max", type=float, default=1.0)
    p.add_argument("--ttt_write_commit_filter_branch_mask", default="0",
                   help="Branches affected by commit risk filter; default branch 0.")
    p.add_argument("--debug_prior_mode",
                   choices=["none", "patch_only", "special_only", "frame_ramp",
                            "reverse_frame_ramp", "checkerboard", "roll"],
                   default="none")
    p.add_argument("--debug_prior_roll_tokens", type=int, default=0)
    p.add_argument("--prior_debug_jsonl", default=None)
    p.add_argument("--hybrid_debug_jsonl", default=None,
                   help="Optional JSONL for per-chunk v2 probe/control state/debug records.")
    p.add_argument("--rho_sem", type=float, default=0.6)
    p.add_argument("--spg_use_g_write_geo", type=int, default=1)

    # Read-path knobs. Identity hooks are used for G2 parity; non-identity
    # controls are intentionally conservative until Phase C.
    p.add_argument("--enable_frame_read_control", type=int, default=0)
    p.add_argument("--enable_swa_read_control", type=int, default=0)
    p.add_argument("--enable_ttt_apply_control", type=int, default=0)
    p.add_argument("--enable_chunk_read_control", type=int, default=0)
    p.add_argument("--beta_frame", type=float, default=0.0)
    p.add_argument("--beta_swa", type=float, default=0.0)
    p.add_argument("--swa_gate_min", type=float, default=0.85)
    p.add_argument("--rho_ttt_apply", type=float, default=0.0)
    p.add_argument("--beta_chunk", type=float, default=0.0)
    p.add_argument("--read_layer_mode",
                   choices=("all", "early", "early_quarter", "early_half", "middle", "late", "single"),
                   default="all")
    p.add_argument("--read_single_layer", type=int, default=-1)
    p.add_argument("--read_cue_source", default="dyn",
                   help="Read cue name. Built-ins include old_dyn/gg.*; ACL2 names such as "
                        "acl2.gg.qq.low.g13_15.off246.headmean.robustq are parsed by HMC.")
    p.add_argument("--read_path", choices=("none", "frame", "swa", "chunk", "ttt_apply"), default="none",
                   help="Phase C v3 convenience selector for one active read path.")
    p.add_argument("--read_topk_frac", type=float, default=0.0)
    p.add_argument("--frame_bias_mode", choices=("pair", "protected_pair", "key", "query"), default="pair")
    p.add_argument("--chunk_bias_mode", choices=("key", "pair"), default="key")
    p.add_argument("--swa_bias_mode", choices=("prev_key",), default="prev_key")
    p.add_argument("--enable_swa_overlap_bias", type=int, default=0,
                   help="Apply an additive SWA attention bias only between current head-overlap queries and previous tail-overlap source tokens.")
    p.add_argument("--swa_overlap_bias_beta", type=float, default=0.0)
    p.add_argument("--swa_overlap_bias_min_keep", type=float, default=1e-4)
    p.add_argument("--swa_overlap_bias_mode",
                   choices=("source", "pair", "union", "intersection"),
                   default="pair")
    p.add_argument("--swa_overlap_bias_layer_mode",
                   choices=("all", "first", "last", "single"),
                   default="last")
    p.add_argument("--swa_overlap_bias_single_layer", type=int, default=-1)
    p.add_argument("--enable_swa_overlap_source_gate", type=int, default=0,
                   help="Gate only previous tail-overlap SWA source tokens using aligned current/previous overlap D maps.")
    p.add_argument("--swa_overlap_source_gate_rho", type=float, default=0.0)
    p.add_argument("--swa_overlap_source_gate_min", type=float, default=0.85)
    p.add_argument("--swa_overlap_source_gate_mode",
                   choices=("source", "prev", "previous", "current", "query", "union",
                            "intersection", "inter", "disagreement", "mismatch",
                            "agree_dyn", "product"),
                   default="source")
    p.add_argument("--swa_overlap_source_gate_target",
                   choices=("v", "value", "k", "key", "kv", "both"),
                   default="v")
    p.add_argument("--swa_overlap_source_gate_layer_mode",
                   choices=("all", "first", "last", "single"),
                   default="last")
    p.add_argument("--swa_overlap_source_gate_single_layer", type=int, default=-1)
    p.add_argument("--enable_swa_overlap_source_replace", type=int, default=0,
                   help="Blend previous tail-overlap SWA source K/V toward aligned current head-overlap K/V for dynamic overlap tokens.")
    p.add_argument("--swa_overlap_source_replace_alpha", type=float, default=0.0)
    p.add_argument("--swa_overlap_source_replace_mode",
                   choices=("source", "prev", "previous", "current", "query", "union", "intersection", "inter", "disagreement", "mismatch", "agree_dyn", "product"),
                   default="union")
    p.add_argument("--swa_overlap_source_replace_target",
                   choices=("v", "value", "k", "key", "kv", "both"),
                   default="kv")
    p.add_argument("--swa_overlap_source_replace_layer_mode",
                   choices=("all", "first", "last", "single"),
                   default="last")
    p.add_argument("--swa_overlap_source_replace_single_layer", type=int, default=-1)
    p.add_argument("--flow_model", choices=("none", "raft", "gmflow", "patch_match"), default="none")
    p.add_argument("--flow_pair_stride", type=int, default=1)
    p.add_argument("--flow_fb_thr", type=float, default=1.5)
    p.add_argument("--flow_residual_thr", type=float, default=0.15)
    p.add_argument("--gram_layer_groups", default="shallow,middle,deep")
    p.add_argument("--read_calib_mode", choices=("none", "per_frame_quantile"), default="none")
    p.add_argument("--read_target_mass", type=float, default=0.06)
    p.add_argument("--read_calib_tau", type=float, default=0.05)
    p.add_argument("--read_blend_lambda", type=float, default=0.25)
    p.add_argument("--read_quality_mass_min", type=float, default=0.03)
    p.add_argument("--read_quality_mass_max", type=float, default=0.20)
    p.add_argument("--read_quality_anchor_max", type=float, default=0.35)
    p.add_argument("--read_quality_frag_max", type=float, default=0.15)
    p.add_argument("--beta_policy", choices=("fixed", "bias_energy_norm"), default="fixed")
    p.add_argument("--beta_energy_target", type=float, default=0.0)
    p.add_argument("--beta_min", type=float, default=0.5)
    p.add_argument("--beta_max", type=float, default=1.5)
    p.add_argument("--stateful_slice_mode", type=int, default=0)
    p.add_argument("--stateful_slice_starts", default="")
    p.add_argument("--stateful_slice_len", type=int, default=0)
    p.add_argument("--save_hmc_states", default=None,
                   help="Directory to save before/after committed HMC states per chunk.")
    p.add_argument("--load_hmc_state_at_chunk", default=None,
                   help="Load a saved HMC state before processing the first local chunk.")
    p.add_argument("--read_protect_ref", type=int, default=1)
    p.add_argument("--read_protect_static", type=int, default=0)
    p.add_argument("--read_protection_mode",
                   choices=("none", "overlap", "anchor", "high_anchor", "static", "reset", "ref",
                            "attention", "attn", "combined_light", "combined_strong"),
                   default="none",
                   help="Phase E read-side protection mask applied to old_dyn/read cue before hook bias.")
    p.add_argument("--read_ref_strength", type=float, default=1.0)
    p.add_argument("--read_overlap_frames", type=int, default=-1,
                   help="Local first/last frames protected by overlap mode; -1 uses --chunk_overlap.")
    p.add_argument("--read_reset_frames", type=int, default=1)
    p.add_argument("--read_attention_q", type=float, default=0.90)
    p.add_argument("--read_static_anchor_thr", type=float, default=0.6)
    p.add_argument("--read_static_dyn_thr", type=float, default=0.3)
    p.add_argument("--hmc_write_score_source",
                   choices=("stage_d", "prior", "bl01", "dyn", "old_dyn", "ttt_residual",
                            "residual", "ttt_residual_reliable", "residual_reliable",
                            "residual_reliability", "alignment_confidence", "alignment",
                            "dg_inv", "dg_locked_inv", "read_inv", "read_inverse", "v5_dg_inv",
                            "explicit_dyn_inv", "exp_dyn_inv", "v5_exp_dyn_inv",
                            "dg_inv_sqrt", "v5_dg_inv_sqrt", "dg_inv_sq", "v5_dg_inv_sq",
                            "stage_d_x_dg_inv", "v5_stage_d_x_dg_inv",
                            "stage_d_x_dg_inv_sqrt", "v5_stage_d_x_dg_inv_sqrt",
                            "stage_d_x_dg_high_inv", "v5_stage_d_x_dg_high_inv",
                            "stage_d_x_dg_high_inv_sqrt", "v5_stage_d_x_dg_high_inv_sqrt",
                            "stage_d_x_exp_inv", "v5_stage_d_x_exp_inv",
                            "stage_d_x_exp_inv_sqrt", "v5_stage_d_x_exp_inv_sqrt",
                            "stage_d_x_exp_inv_sq", "v5_stage_d_x_exp_inv_sq",
                            "stage_d_x_exp_focal2",
                            "stage_d_x_dg_exp_inv_sqrt", "v5_stage_d_x_dg_exp_inv_sqrt",
                            "stage_d_x_dg_exp_inter_inv", "v5_stage_d_x_dg_exp_inter_inv",
                            "stage_d_x_dg_exp_inter_inv_sqrt", "v5_stage_d_x_dg_exp_inter_inv_sqrt",
                            "stage_d_x_union_dyn_inv", "v5_stage_d_x_union_dyn_inv",
                            "stage_d_x_dg_inv_sq", "v5_stage_d_x_dg_inv_sq",
                            "stage_d_x_static_focal2",
                            "stage_d_x_dg_inv_pow4", "v5_stage_d_x_dg_inv_pow4",
                            "stage_d_x_static_focal4",
                            "stage_d_x_dg_boundary_focal2", "v5_stage_d_x_dg_boundary_focal2"),
                   default="stage_d",
                   help="Phase E optional override for the explicit probe-cache TTT write prior.")
    p.add_argument("--hmc_write_sparse_ratio", type=float, default=1.0)
    p.add_argument("--hmc_write_sparse_mode",
                   choices=("none", "hard", "exact", "exact_preserve", "soft"),
                   default="none")
    p.add_argument("--ttt_apply_min_gate", type=float, default=0.0)
    p.add_argument("--enable_swa_write_control", type=int, default=0)
    p.add_argument(
        "--swa_write_mode",
        choices=(
            "none", "v", "value", "k", "key", "kv", "both",
            "v_centered", "kv_centered",
            "v_resid", "kv_resid", "v_resid_centered", "kv_resid_centered",
        ),
        default="none",
    )
    p.add_argument("--swa_write_rho", type=float, default=0.0)
    p.add_argument("--swa_write_min_gate", type=float, default=0.0)
    p.add_argument("--swa_write_sparse_ratio", type=float, default=1.0)
    p.add_argument("--swa_write_layer_mode",
                   choices=("all", "first", "last", "early", "middle", "late", "single"),
                   default="all")
    p.add_argument("--swa_write_single_layer", type=int, default=-1)
    p.add_argument("--swa_write_scope",
                   choices=("all", "tail_overlap", "head_overlap", "both_overlap"),
                   default="all")
    p.add_argument("--swa_write_keep_scope",
                   choices=(
                       "all",
                       "tail_overlap", "head_overlap", "both_overlap",
                       "exclude_tail_overlap", "exclude_head_overlap", "exclude_both_overlap",
                   ),
                   default="all",
                   help="Optionally keep only an overlap slice, or exclude an overlap slice, in committed SWA history.")
    p.add_argument("--swa_write_score_source",
                   choices=("read", "dg", "explicit_dyn", "old_dyn", "union_dyn", "intersection"),
                   default="read",
                   help="Dynamic score used for SWA write-side K/V gating; read means D_g, union_dyn=max(D_g, explicit_dyn).")
    p.add_argument("--swa_write_cache_blend_alpha", type=float, default=0.0,
                   help="Blend committed SWA history K/V from pre-SWA cache toward post-SWA cache in the selected write scope.")
    p.add_argument("--swa_write_cache_blend_mode",
                   choices=("dynamic", "static", "all"),
                   default="dynamic",
                   help="Score used for post-SWA cache blend: dynamic=D, static=1-D, all=constant alpha.")
    p.add_argument("--swa_write_cache_blend_target",
                   choices=("v", "value", "k", "key", "kv", "both"),
                   default="v",
                   help="Which SWA history tensors receive post-cache blending.")
    p.add_argument("--ttt_write_token_scope",
                   choices=(
                       "all",
                       "tail_overlap", "head_overlap", "both_overlap",
                       "tail_overlap_veto", "head_overlap_veto", "both_overlap_veto",
                       "tail_overlap_drop", "head_overlap_drop", "both_overlap_drop",
                       "tail_overlap_native", "head_overlap_native", "both_overlap_native",
                       "tail_overlap_no_boost", "head_overlap_no_boost", "both_overlap_no_boost",
                   ),
                   default="all",
                   help="Optionally restrict or protect overlap-frame TTT replay prior; *_veto keeps non-overlap native and applies prior only inside overlap; *_drop suppresses the overlap seam; *_native/no_boost preserve full replay outside overlap while protecting overlap continuity.")
    p.add_argument("--ttt_write_token_scope_floor", type=float, default=0.0,
                   help="Multiplier for tokens outside --ttt_write_token_scope; 0.0 is hard scope, 0.25 keeps weak non-overlap replay.")
    p.add_argument("--fast_cue_eval", type=int, default=0,
                   help="Skip unused cue-family CPU statistics for simple ACL2 headmean sources.")

    # -- Video output -----------------------------------------------------
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mask_alpha", type=float, default=0.40)
    return p


def _resolve_hybrid_mode(args: argparse.Namespace) -> str:
    if args.hybrid_memory_mode is not None:
        return args.hybrid_memory_mode
    return {
        "semantic": "ttt_write_only",
        "unity_replay": "unity_replay",
        "native": "native",
    }[args.ttt_write_mode]


def _validate_args(args: argparse.Namespace, mode: str) -> None:
    if args.config and not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")
    if args.geometry_eval_mode and mode in {"ttt_write_only", "hybrid", "read_path_only"}:
        sys.exit(f"--geometry_eval_mode skips Stage B/D and cannot run hybrid_memory_mode={mode}; use unity_replay/native/probe_only.")
    if mode in {"ttt_write_only", "hybrid", "read_path_only"} and args.stage_c_mode != "none":
        if args.sam_backend == "sam2":
            if not args.sam2_checkpoint or not os.path.isfile(args.sam2_checkpoint):
                sys.exit(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
            if not args.sam2_model_cfg:
                sys.exit("sam2_model_cfg is required when --sam_backend=sam2.")
        elif args.sam_backend == "sam3":
            if not args.sam3_checkpoint or not os.path.isfile(args.sam3_checkpoint):
                sys.exit(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")
        elif args.sam_backend == "sam31_multiplex":
            if args.sam31_checkpoint and not os.path.isfile(args.sam31_checkpoint):
                sys.exit(f"SAM3.1 checkpoint not found: {args.sam31_checkpoint}")


def _apply_read_path_selector(args: argparse.Namespace) -> None:
    """Map Phase C v3's single read-path selector onto existing booleans."""

    if args.read_path == "none":
        return
    args.enable_frame_read_control = int(args.read_path == "frame")
    args.enable_swa_read_control = int(args.read_path == "swa")
    args.enable_chunk_read_control = int(args.read_path == "chunk")
    args.enable_ttt_apply_control = int(args.read_path == "ttt_apply")


def _build_stage_b(args: argparse.Namespace) -> DynamicCueExtractor:
    return DynamicCueExtractor(
        k_intra=args.k_intra,
        use_attention_prior=not args.disable_attention_prior,
        support_time_decay=args.support_time_decay,
        support_temporal_weight=args.support_temporal_weight,
        support_affinity_weight=args.support_affinity_weight,
        support_static_weight=args.support_static_weight,
        sigma_pt=args.sigma_pt,
        proxy_mode=args.stageb_proxy_mode,
        tau_occ=args.tau_occ,
        alpha_1=args.alpha_1,
        alpha_3=args.alpha_3,
        attn_stat_fusion_weight=args.attn_stat_fusion_weight,
        dyn_fusion_mode=args.dyn_fusion_mode,
        implicit_weight=args.implicit_weight,
        implicit_gate_floor=args.implicit_gate_floor,
        implicit_calib_min_range=args.implicit_calib_min_range,
        nonocc_dynamic=bool(args.stageb_nonocc_dynamic),
        lambda_s=args.lambda_s,
        lambda_a=args.lambda_a,
        lambda_d=args.lambda_d,
        lambda_o=args.lambda_o,
        lambda_u=args.lambda_u,
    )


def _build_stage_c_kwargs(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    frontend_kwargs: Dict[str, Any] = dict(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        ann_frame_idx=args.ann_frame_idx,
        discovery_frame_stride=args.discovery_frame_stride,
        max_thing_objects=args.max_thing_objects,
    )
    if args.thing_prompts:
        frontend_kwargs["thing_prompts"] = [s.strip() for s in args.thing_prompts.split(",")]
    if args.stuff_prompts:
        frontend_kwargs["stuff_prompts"] = [s.strip() for s in args.stuff_prompts.split(",")]

    build_kwargs: Dict[str, Any] = dict(
        sam_backend=args.sam_backend,
        sam3_checkpoint=args.sam3_checkpoint,
        sam31_checkpoint=args.sam31_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_model_cfg=args.sam2_model_cfg,
        detector_type=args.detector,
        gdino_config=args.gdino_config,
        gdino_checkpoint=args.gdino_checkpoint,
        yoloe_model=args.yoloe_model,
        yoloe_batch_size=args.yoloe_batch_size,
        yoloe_imgsz=args.yoloe_imgsz,
        sam31_offload_video_to_cpu=args.sam31_offload_video_to_cpu,
        sam31_offload_outputs_to_cpu=args.sam31_offload_outputs_to_cpu,
        sam31_offload_sam_during_detection=args.sam31_offload_sam_during_detection,
        sam31_text_track_labels=args.sam31_text_track_labels,
        sam31_direct_text_prompt_labels=args.sam31_direct_text_prompt_labels,
        sam31_direct_text_prompt_frame_count=args.sam31_direct_text_prompt_frame_count,
        sam31_structure_prompt_labels=args.sam31_structure_prompt_labels,
        sam31_structure_prompt_frame_count=args.sam31_structure_prompt_frame_count,
        sam31_structure_prompt_chunk_stride=args.sam31_structure_prompt_chunk_stride,
        sam31_person_refresh_prompt_frames=args.sam31_person_refresh_prompt_frames,
        sam31_nontext_object_prompt_budget=args.sam31_nontext_object_prompt_budget,
        sam31_nontext_object_prompt_min_support=args.sam31_nontext_object_prompt_min_support,
        sam31_text_object_prompt_budget=args.sam31_text_object_prompt_budget,
        sam31_nontext_sparse_support=bool(args.sam31_nontext_sparse_support),
        sam31_max_text_prompt_objects=args.sam31_max_text_prompt_objects,
        sam31_max_internal_objects=args.sam31_max_internal_objects,
        sam31_max_movable_objects=args.sam31_max_movable_objects,
        sam31_max_static_objects=args.sam31_max_static_objects,
        sam31_max_structure_objects=args.sam31_max_structure_objects,
        device=args.device,
    )
    return build_kwargs, frontend_kwargs


def _reset_hybrid_state_if_needed(state: Optional[HybridMemoryState]) -> Optional[HybridMemoryState]:
    if state is None or state.ttt_state is None:
        return state
    ttt = state.ttt_state
    preserved_history = ttt.get("history")
    reset_state: Dict[str, Any] = {
        "w0": [None] * len(ttt.get("w0", [])),
        "w1": [None] * len(ttt.get("w1", [])),
        "w2": [None] * len(ttt.get("w2", [])),
    }
    if preserved_history is not None:
        reset_state["history"] = preserved_history
    return HybridMemoryState(
        ttt_state=reset_state,
        swa_state=state.swa_state,
        ref_state=state.ref_state,
        prev_control_summary=state.prev_control_summary,
        debug=dict(state.debug),
    )


def _parse_chunk_index_set(value: str) -> set[int]:
    out: set[int] = set()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.add(int(part))
    return out


def _parse_chunk_float_map(value: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        sep = ":" if ":" in part else "=" if "=" in part else None
        if sep is None:
            raise ValueError(
                f"Expected CHUNK:SCALE in --ttt_semantic_write_scale_chunks, got {part!r}"
            )
        chunk_s, scale_s = part.split(sep, 1)
        out[int(chunk_s.strip())] = float(scale_s.strip())
    return out


def _scale_fast_weight_delta_and_renorm(
    base: torch.Tensor,
    candidate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    base_on_candidate = base.to(device=candidate.device, dtype=candidate.dtype)
    out = base_on_candidate + float(scale) * (candidate - base_on_candidate)
    if out.ndim >= 2:
        base_norm = base_on_candidate.detach().norm(dim=1, keepdim=True)
        out_norm = out.norm(dim=1, keepdim=True)
        out = out / (out_norm + 1e-5) * base_norm
    return out


def _blend_semantic_ttt_against_native(
    native_ttt: Optional[Dict[str, Any]],
    semantic_ttt: Optional[Dict[str, Any]],
    scale: float,
) -> Optional[Dict[str, Any]]:
    if native_ttt is None or semantic_ttt is None:
        return semantic_ttt
    blended: Dict[str, Any] = dict(semantic_ttt)
    for branch in ("w0", "w1", "w2"):
        native_list = native_ttt.get(branch)
        semantic_list = semantic_ttt.get(branch)
        if not isinstance(native_list, list) or not isinstance(semantic_list, list):
            continue
        out_list: List[Any] = []
        for native_w, semantic_w in zip(native_list, semantic_list):
            if isinstance(native_w, torch.Tensor) and isinstance(semantic_w, torch.Tensor):
                out_list.append(_scale_fast_weight_delta_and_renorm(native_w, semantic_w, scale))
            else:
                out_list.append(semantic_w)
        if len(semantic_list) > len(out_list):
            out_list.extend(semantic_list[len(out_list):])
        blended[branch] = out_list
    # Preserve the semantic/SWA-gated history; only fast-weight write deltas are scaled.
    return blended


def _save_hmc_state(path: Path, state: HybridMemoryState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_cpu = HybridMemoryState(
        ttt_state=_move_tensor_tree_to_device(state.ttt_state, "cpu") if state.ttt_state is not None else None,
        swa_state=_move_tensor_tree_to_device(state.swa_state, "cpu") if state.swa_state is not None else None,
        ref_state=_move_tensor_tree_to_device(state.ref_state, "cpu") if state.ref_state is not None else None,
        prev_control_summary=_move_tensor_tree_to_device(state.prev_control_summary, "cpu")
        if state.prev_control_summary is not None else None,
        debug=dict(state.debug),
    )
    torch.save(state_cpu, path)


def _load_hmc_state(path: str) -> HybridMemoryState:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, HybridMemoryState):
        return state
    if isinstance(state, dict):
        return HybridMemoryState(**state)
    raise TypeError(f"Unsupported HMC state file payload: {type(state)!r}")


def _infer_run_dir(args: argparse.Namespace) -> Optional[Path]:
    for value in (args.output_txt, args.output_pt, args.hybrid_debug_jsonl, args.prior_debug_jsonl):
        if value:
            return Path(value).parent
    if args.output_video:
        return Path(args.output_video).parent
    return None


def _write_hmc_config(args: argparse.Namespace, mode: str) -> None:
    out_dir = _infer_run_dir(args)
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(vars(args))
    payload["resolved_hybrid_memory_mode"] = mode
    payload["two_pass"] = True
    payload["commit_source"] = args.hmc_commit_mode
    payload["read_path_hooks_status"] = "real_identity_hook_sites"
    with open(out_dir / "hmc_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=True)
    hook_identity = {
        "status": "active" if mode == "identity_hooks" else "available",
        "reason": "Read-path hooks are wired at real model sites; identity mode records coverage without modifying tensors.",
        "frame_attention_hook": "implemented_identity_passthrough",
        "swa_read_hook": "implemented_identity_passthrough",
        "ttt_apply_hook": "implemented_identity_passthrough",
        "chunk_attention_hook": "implemented_identity_passthrough",
        "ttt_update_replay": "implemented",
        "identity_ttt_update": mode in {"unity_replay", "read_path_only"},
        "native_ttt_update": mode in {"native", "identity_hooks"},
        "hmc_commit_mode": args.hmc_commit_mode,
    }
    with open(out_dir / "hmc_hook_identity_check.json", "w", encoding="utf-8") as f:
        json.dump(hook_identity, f, indent=2, sort_keys=True)


def _rotation_angle_deg(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    same = (R_a - R_b).abs().amax(dim=(-1, -2)) < 1e-7
    rel = torch.matmul(R_a.transpose(-1, -2), R_b)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    return torch.where(same, torch.zeros_like(angle), angle)


def _geometry_diff_summary(pass1: GeometryOutput, pass2: Optional[GeometryOutput]) -> Dict[str, Any]:
    if pass2 is None:
        return {}
    out: Dict[str, Any] = {}
    T = min(int(pass1.camera_poses.shape[0]), int(pass2.camera_poses.shape[0]))
    if T > 0:
        p1 = pass1.camera_poses[:T].detach().cpu().float()
        p2 = pass2.camera_poses[:T].detach().cpu().float()
        t_diff = torch.linalg.norm(p1[:, :3, 3] - p2[:, :3, 3], dim=-1)
        r_diff = _rotation_angle_deg(p1[:, :3, :3], p2[:, :3, :3])
        out.update({
            "pass1_pass2_pose_t_mean": float(t_diff.mean().item()),
            "pass1_pass2_pose_t_max": float(t_diff.max().item()),
            "pass1_pass2_pose_r_deg_mean": float(r_diff.mean().item()),
            "pass1_pass2_pose_r_deg_max": float(r_diff.max().item()),
            "pass1_pass2_pose_matrix_abs_max": float((p1 - p2).abs().max().item()),
        })

    def _tensor_l1_stats(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
        if a is None or b is None:
            return
        common_shape = tuple(min(int(sa), int(sb)) for sa, sb in zip(a.shape, b.shape))
        if not common_shape or any(s <= 0 for s in common_shape):
            return
        slices = tuple(slice(0, s) for s in common_shape)
        da = a[slices].detach().cpu().float()
        db = b[slices].detach().cpu().float()
        diff = (da - db).abs()
        out[f"pass1_pass2_{name}_l1_mean"] = float(diff.mean().item())
        out[f"pass1_pass2_{name}_abs_max"] = float(diff.max().item())

    _tensor_l1_stats("local_points", pass1.local_points, pass2.local_points)
    _tensor_l1_stats("world_points", pass1.world_points, pass2.world_points)
    _tensor_l1_stats("confidence", pass1.confidence, pass2.confidence)
    return out


def _iter_named_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(obj):
        out[prefix or "tensor"] = obj.detach().cpu().float()
    elif isinstance(obj, dict):
        for key in sorted(obj.keys()):
            child = obj[key]
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_iter_named_tensors(child, child_prefix))
    elif isinstance(obj, (list, tuple)):
        for idx, child in enumerate(obj):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            out.update(_iter_named_tensors(child, child_prefix))
    return out


def _tensor_tree_rel_diff(a: Any, b: Any) -> Dict[str, Any]:
    ta = _iter_named_tensors(a)
    tb = _iter_named_tensors(b)
    common = sorted(set(ta).intersection(tb))
    rel_vals: List[float] = []
    abs_vals: List[float] = []
    for key in common:
        aa = ta[key]
        bb = tb[key]
        if aa.shape != bb.shape:
            continue
        diff = torch.linalg.norm(aa - bb).item()
        denom = torch.linalg.norm(bb).item() + 1e-8
        rel_vals.append(float(diff / denom))
        abs_vals.append(float(diff))
    return {
        "tensor_count_a": len(ta),
        "tensor_count_b": len(tb),
        "common_tensor_count": len(common),
        "mean_rel_diff": float(sum(rel_vals) / len(rel_vals)) if rel_vals else 0.0,
        "max_rel_diff": float(max(rel_vals)) if rel_vals else 0.0,
        "mean_abs_diff": float(sum(abs_vals) / len(abs_vals)) if abs_vals else 0.0,
        "max_abs_diff": float(max(abs_vals)) if abs_vals else 0.0,
    }


def _state_side_effect_summary(
    *,
    controlled: Optional[HybridMemoryState],
    probe_native: Optional[HybridMemoryState],
) -> Dict[str, Any]:
    if controlled is None or probe_native is None:
        return {}
    out: Dict[str, Any] = {
        "controlled_hash": hybrid_state_fingerprint(controlled),
        "probe_native_hash": hybrid_state_fingerprint(probe_native),
        "hash_equal": hybrid_state_fingerprint(controlled) == hybrid_state_fingerprint(probe_native),
    }
    out["ttt_state_diff"] = _tensor_tree_rel_diff(controlled.ttt_state, probe_native.ttt_state)
    out["swa_state_diff"] = _tensor_tree_rel_diff(controlled.swa_state, probe_native.swa_state)
    out["ref_state_diff"] = _tensor_tree_rel_diff(controlled.ref_state, probe_native.ref_state)
    for branch in ("w0", "w1", "w2"):
        controlled_branch = (controlled.ttt_state or {}).get(branch)
        probe_branch = (probe_native.ttt_state or {}).get(branch)
        out[f"ttt_{branch}_diff"] = _tensor_tree_rel_diff(controlled_branch, probe_branch)
    return out


def _append_hybrid_debug_jsonl(
    path: Optional[str],
    *,
    args: argparse.Namespace,
    mode: str,
    chunk_idx: int,
    start: int,
    end: int,
    probe: ProbeOutput,
    probe_after_state_hash: str,
    control_prior: Optional[HybridMemoryControlPrior],
    result: Optional[HybridMemoryResult],
    committed_state_next: Optional[HybridMemoryState] = None,
    memory_side_effect: Optional[Dict[str, Any]] = None,
    geometry_diff: Optional[Dict[str, Any]] = None,
) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rec: Dict[str, Any] = {
        "chunk_idx": int(chunk_idx),
        "start_frame": int(start),
        "end_frame": int(end),
        "hybrid_memory_mode": mode,
        "hmc_commit_mode": args.hmc_commit_mode,
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "hash_H_m_before_probe": probe.debug.get("committed_state_hash"),
        "hash_H_m_after_probe": probe_after_state_hash,
        "hash_H_m_before_pass2": probe.debug.get("committed_state_hash"),
        "probe_committed_state_hash": probe.debug.get("committed_state_hash"),
        "probe_native_provisional_state_hash": probe.debug.get("native_provisional_state_hash"),
        "probe_ttt_cache_layers": probe.debug.get("ttt_cache_layers"),
        "probe_token_count": probe.debug.get("token_count"),
        "probe_no_commit_hash_equal": probe_after_state_hash == probe.debug.get("committed_state_hash"),
        "read_path_controls_requested": {
            "frame": bool(args.enable_frame_read_control),
            "swa": bool(args.enable_swa_read_control),
            "ttt_apply": bool(args.enable_ttt_apply_control),
            "chunk": bool(args.enable_chunk_read_control),
            "beta_frame": float(args.beta_frame),
            "beta_swa": float(args.beta_swa),
            "rho_ttt_apply": float(args.rho_ttt_apply),
            "beta_chunk": float(args.beta_chunk),
            "read_layer_mode": args.read_layer_mode,
            "read_single_layer": int(args.read_single_layer),
            "read_cue_source": args.read_cue_source,
            "read_path": args.read_path,
            "read_topk_frac": float(args.read_topk_frac),
            "frame_bias_mode": args.frame_bias_mode,
            "chunk_bias_mode": args.chunk_bias_mode,
            "swa_bias_mode": args.swa_bias_mode,
            "flow_model": args.flow_model,
            "flow_pair_stride": int(args.flow_pair_stride),
            "flow_fb_thr": float(args.flow_fb_thr),
            "flow_residual_thr": float(args.flow_residual_thr),
            "gram_layer_groups": args.gram_layer_groups,
            "read_calib_mode": args.read_calib_mode,
            "read_target_mass": float(args.read_target_mass),
            "read_calib_tau": float(args.read_calib_tau),
            "read_blend_lambda": float(args.read_blend_lambda),
            "read_quality_mass_min": float(args.read_quality_mass_min),
            "read_quality_mass_max": float(args.read_quality_mass_max),
            "read_quality_anchor_max": float(args.read_quality_anchor_max),
            "read_quality_frag_max": float(args.read_quality_frag_max),
            "beta_policy": args.beta_policy,
            "beta_energy_target": float(args.beta_energy_target),
            "beta_min": float(args.beta_min),
            "beta_max": float(args.beta_max),
            "read_protect_ref": bool(args.read_protect_ref),
            "read_protect_static": bool(args.read_protect_static),
            "read_protection_mode": args.read_protection_mode,
            "read_ref_strength": float(args.read_ref_strength),
            "read_overlap_frames": int(args.chunk_overlap if int(args.read_overlap_frames) < 0 else args.read_overlap_frames),
            "read_reset_frames": int(args.read_reset_frames),
            "read_attention_q": float(args.read_attention_q),
            "hmc_write_score_source": args.hmc_write_score_source,
            "hmc_write_sparse_ratio": float(args.hmc_write_sparse_ratio),
            "hmc_write_sparse_mode": args.hmc_write_sparse_mode,
            "ttt_apply_min_gate": float(args.ttt_apply_min_gate),
            "fast_cue_eval": bool(args.fast_cue_eval),
        },
        "probe_hmc_hook_trace_counts": probe.debug.get("hmc_hook_trace_counts"),
    }
    if control_prior is not None:
        rec.update({
            "prior_mean_D_tok": control_prior.debug.get("mean_D_tok"),
            "prior_max_D_tok": control_prior.debug.get("max_D_tok"),
            "prior_q90_D_tok": control_prior.debug.get("q90_D_tok"),
            "prior_mean_D_patch": control_prior.debug.get("mean_D_patch"),
            "prior_q10_D_patch": control_prior.debug.get("q10_D_patch"),
            "prior_q50_D_patch": control_prior.debug.get("q50_D_patch"),
            "prior_q90_D_patch": control_prior.debug.get("q90_D_patch"),
            "prior_dynamic_mass_D_gt_050": control_prior.debug.get("dynamic_mass_D_gt_050"),
            "prior_dynamic_mass_D_gt_075": control_prior.debug.get("dynamic_mass_D_gt_075"),
            "prior_dynamic_mass_D_gt_001": control_prior.debug.get("dynamic_mass_D_gt_001"),
            "prior_anchor_collision": control_prior.debug.get("anchor_collision"),
            "prior_fragmentation": control_prior.debug.get("fragmentation"),
            "prior_old_dyn_iou": control_prior.debug.get("old_dyn_iou"),
            "prior_old_dyn_coverage": control_prior.debug.get("old_dyn_coverage"),
            "prior_old_dyn_recall": control_prior.debug.get("old_dyn_recall"),
            "prior_cue_quality_pass": control_prior.debug.get("cue_quality_pass"),
            "prior_cue_quality_mass_pass": control_prior.debug.get("cue_quality_mass_pass"),
            "prior_cue_quality_anchor_pass": control_prior.debug.get("cue_quality_anchor_pass"),
            "prior_cue_quality_frag_pass": control_prior.debug.get("cue_quality_frag_pass"),
            "prior_cue_gate": control_prior.debug.get("cue_gate"),
            "prior_fallback_rate": control_prior.debug.get("fallback_rate"),
            "prior_cue_source_effective": control_prior.debug.get("cue_source_effective"),
            "prior_corr_D_unc": control_prior.debug.get("corr_D_unc"),
            "prior_corr_D_occ": control_prior.debug.get("corr_D_occ"),
            "prior_corr_D_conf": control_prior.debug.get("corr_D_conf"),
            "prior_corr_D_inv_conf": control_prior.debug.get("corr_D_inv_conf"),
            "prior_corr_D_old_dyn": control_prior.debug.get("corr_D_old_dyn"),
            "prior_mean_R_tok": control_prior.debug.get("mean_R_tok"),
            "prior_read_cue_source": control_prior.debug.get("read_cue_source"),
            "prior_read_topk_frac": control_prior.debug.get("read_topk_frac"),
            "prior_frame_bias_mode": control_prior.debug.get("frame_bias_mode"),
            "prior_chunk_bias_mode": control_prior.debug.get("chunk_bias_mode"),
            "prior_swa_bias_mode": control_prior.debug.get("swa_bias_mode"),
            "prior_flow_model": control_prior.debug.get("flow_model"),
            "prior_gram_layer_groups": control_prior.debug.get("gram_layer_groups"),
            "prior_read_calib_mode": control_prior.debug.get("read_calib_mode"),
            "prior_read_target_mass": control_prior.debug.get("read_target_mass"),
            "prior_read_calib_tau": control_prior.debug.get("read_calib_tau"),
            "prior_read_blend_lambda": control_prior.debug.get("read_blend_lambda"),
            "prior_beta_policy": control_prior.debug.get("beta_policy"),
            "prior_beta_frame_effective": control_prior.debug.get("beta_frame_effective"),
            "prior_beta_energy_target": control_prior.debug.get("beta_energy_target"),
            "prior_beta_raw_frame_bias_energy": control_prior.debug.get("beta_raw_frame_bias_energy"),
            "prior_beta_was_clipped": control_prior.debug.get("beta_was_clipped"),
            "prior_protected_token_count": control_prior.debug.get("protected_token_count"),
            "prior_safe_patch_token_count": control_prior.debug.get("safe_patch_token_count"),
            "prior_read_protection_mode": control_prior.debug.get("read_protection_mode"),
            "prior_read_ref_strength": control_prior.debug.get("read_ref_strength"),
            "prior_protect_patch_count": control_prior.debug.get("protect_patch_count"),
            "prior_protect_patch_mass": control_prior.debug.get("protect_patch_mass"),
            "prior_protect_overlap_count": control_prior.debug.get("protect_overlap_count"),
            "prior_protect_anchor_count": control_prior.debug.get("protect_anchor_count"),
            "prior_protect_attention_count": control_prior.debug.get("protect_attention_count"),
            "prior_hmc_write_score_source": control_prior.debug.get("hmc_write_score_source"),
            "prior_hmc_write_override": control_prior.debug.get("hmc_write_override"),
            "prior_hmc_write_sparse_ratio": control_prior.debug.get("hmc_write_sparse_ratio"),
            "prior_hmc_write_selected_mass": control_prior.debug.get("hmc_write_selected_mass"),
            "prior_hmc_write_score_mean": control_prior.debug.get("hmc_write_score_mean"),
            "prior_hmc_write_residual_mean_patch": control_prior.debug.get("hmc_write_residual_mean_patch"),
            "prior_hmc_write_reliability_mean_patch": control_prior.debug.get("hmc_write_reliability_mean_patch"),
            "prior_hmc_write_corr_score_dyn": control_prior.debug.get("hmc_write_corr_score_dyn"),
            "prior_hmc_write_corr_score_exp_dyn": control_prior.debug.get("hmc_write_corr_score_exp_dyn"),
            "prior_hmc_write_corr_score_unc": control_prior.debug.get("hmc_write_corr_score_unc"),
            "prior_ttt_write_present": control_prior.debug.get("ttt_write_prior_present"),
            "prior_ttt_write_mean": control_prior.debug.get("ttt_write_prior_mean"),
            "prior_read_path_control": control_prior.debug.get("read_path_control"),
        })
    if result is not None:
        rec.update({
            "controlled_input_state_hash": result.debug.get("controlled_input_state_hash"),
            "controlled_output_state_hash": result.debug.get("controlled_output_state_hash"),
            "hash_H_m_after_commit": hybrid_state_fingerprint(committed_state_next) if committed_state_next is not None else result.debug.get("controlled_output_state_hash"),
            "hash_H_next": hybrid_state_fingerprint(committed_state_next) if committed_state_next is not None else result.debug.get("controlled_output_state_hash"),
            "commit_source_state_hash": hybrid_state_fingerprint(committed_state_next) if committed_state_next is not None else result.debug.get("controlled_output_state_hash"),
            "state_double_write_safe": (
                result.debug.get("controlled_input_state_hash")
                == probe.debug.get("committed_state_hash")
            ),
            "control_trace": result.control_trace,
        })
    if memory_side_effect:
        rec["memory_side_effect"] = memory_side_effect
        ttt_diff = memory_side_effect.get("ttt_state_diff") or {}
        rec["memory_ttt_mean_rel_diff"] = ttt_diff.get("mean_rel_diff")
        rec["memory_ttt_max_rel_diff"] = ttt_diff.get("max_rel_diff")
        for branch in ("w0", "w1", "w2"):
            branch_diff = memory_side_effect.get(f"ttt_{branch}_diff") or {}
            rec[f"memory_ttt_{branch}_mean_rel_diff"] = branch_diff.get("mean_rel_diff")
            rec[f"memory_ttt_{branch}_max_rel_diff"] = branch_diff.get("max_rel_diff")
    if geometry_diff:
        rec.update(geometry_diff)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")

    out_dir = Path(path).parent
    probe_summary = {
        k: rec.get(k)
        for k in (
            "chunk_idx", "start_frame", "end_frame", "hybrid_memory_mode",
            "hash_H_m_before_probe", "hash_H_m_after_probe",
            "probe_native_provisional_state_hash", "probe_no_commit_hash_equal",
            "probe_ttt_cache_layers", "probe_token_count",
        )
    }
    control_summary = {
        k: rec.get(k)
        for k in (
            "chunk_idx", "start_frame", "end_frame", "hybrid_memory_mode",
            "hash_H_m_before_pass2", "hash_H_m_after_commit", "hash_H_next",
            "hmc_commit_mode", "state_double_write_safe", "pass1_pass2_pose_t_mean",
            "pass1_pass2_pose_t_max", "pass1_pass2_pose_matrix_abs_max",
            "pass1_pass2_local_points_l1_mean", "pass1_pass2_world_points_l1_mean",
            "memory_ttt_mean_rel_diff", "memory_ttt_max_rel_diff",
            "memory_ttt_w0_mean_rel_diff", "memory_ttt_w1_mean_rel_diff", "memory_ttt_w2_mean_rel_diff",
            "control_trace",
        )
    }
    with open(out_dir / "hmc_probe_summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(probe_summary, sort_keys=True) + "\n")
    with open(out_dir / "hmc_control_summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(control_summary, sort_keys=True) + "\n")
    if result is not None:
        hook_effect = {
            "chunk_idx": int(chunk_idx),
            "start_frame": int(start),
            "end_frame": int(end),
            "hybrid_memory_mode": mode,
            "hook_effect_summary": result.control_trace.get("hook_effect_summary", {}),
            "hook_trace_counts": result.control_trace.get("hook_trace_counts", {}),
            "implemented_paths": result.control_trace.get("implemented_paths", []),
        }
        with open(out_dir / "hook_effect_summary.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(hook_effect, sort_keys=True) + "\n")
    if control_prior is not None:
        cue_quality = {
            k: rec.get(k)
            for k in (
                "chunk_idx", "start_frame", "end_frame", "hybrid_memory_mode",
                "prior_read_cue_source", "prior_mean_D_patch", "prior_q10_D_patch",
                "prior_q50_D_patch", "prior_q90_D_patch",
                "prior_dynamic_mass_D_gt_001",
                "prior_dynamic_mass_D_gt_050", "prior_dynamic_mass_D_gt_075",
                "prior_anchor_collision", "prior_fragmentation",
                "prior_old_dyn_iou", "prior_old_dyn_coverage", "prior_old_dyn_recall",
                "prior_cue_quality_pass", "prior_cue_quality_mass_pass",
                "prior_cue_quality_anchor_pass", "prior_cue_quality_frag_pass",
                "prior_cue_gate", "prior_fallback_rate", "prior_cue_source_effective",
                "prior_corr_D_unc", "prior_corr_D_occ", "prior_corr_D_conf",
                "prior_corr_D_inv_conf", "prior_corr_D_old_dyn",
                "prior_protected_token_count", "prior_safe_patch_token_count",
                "prior_flow_model", "prior_gram_layer_groups",
                "prior_read_calib_mode", "prior_read_target_mass",
                "prior_read_calib_tau", "prior_read_blend_lambda",
                "prior_beta_policy", "prior_beta_frame_effective",
                "prior_beta_energy_target", "prior_beta_raw_frame_bias_energy",
                "prior_beta_was_clipped",
                "prior_read_protection_mode", "prior_read_ref_strength",
                "prior_protect_patch_count", "prior_protect_patch_mass",
                "prior_protect_overlap_count", "prior_protect_anchor_count",
                "prior_protect_attention_count",
                "prior_hmc_write_score_source", "prior_hmc_write_override",
                "prior_hmc_write_sparse_ratio", "prior_hmc_write_selected_mass",
                "prior_hmc_write_score_mean", "prior_hmc_write_residual_mean_patch",
                "prior_hmc_write_reliability_mean_patch",
                "prior_hmc_write_corr_score_dyn", "prior_hmc_write_corr_score_exp_dyn",
                "prior_hmc_write_corr_score_unc",
            )
        }
        with open(out_dir / "cue_quality_per_chunk.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(cue_quality, sort_keys=True) + "\n")


def _write_cue_quality_summary(args: argparse.Namespace) -> None:
    out_dir = _infer_run_dir(args)
    if out_dir is None:
        return
    path = out_dir / "cue_quality_per_chunk.jsonl"
    if not path.is_file():
        return
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return
    numeric_keys = [
        "prior_mean_D_patch",
        "prior_q10_D_patch",
        "prior_q50_D_patch",
        "prior_q90_D_patch",
        "prior_dynamic_mass_D_gt_001",
        "prior_dynamic_mass_D_gt_050",
        "prior_dynamic_mass_D_gt_075",
        "prior_anchor_collision",
        "prior_fragmentation",
        "prior_old_dyn_iou",
        "prior_old_dyn_coverage",
        "prior_old_dyn_recall",
        "prior_cue_gate",
        "prior_fallback_rate",
        "prior_beta_frame_effective",
        "prior_beta_raw_frame_bias_energy",
        "prior_corr_D_unc",
        "prior_corr_D_occ",
        "prior_corr_D_conf",
        "prior_corr_D_inv_conf",
        "prior_corr_D_old_dyn",
    ]
    summary: Dict[str, Any] = {
        "num_chunks": len(rows),
        "read_cue_source": rows[0].get("prior_read_cue_source"),
        "flow_model": rows[0].get("prior_flow_model"),
        "gram_layer_groups": rows[0].get("prior_gram_layer_groups"),
        "read_calib_mode": rows[0].get("prior_read_calib_mode"),
        "read_target_mass": rows[0].get("prior_read_target_mass"),
        "read_calib_tau": rows[0].get("prior_read_calib_tau"),
        "read_blend_lambda": rows[0].get("prior_read_blend_lambda"),
    }
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None]
        if vals:
            tensor_vals = torch.tensor(vals, dtype=torch.float32)
            summary[f"mean_{key}"] = float(tensor_vals.mean().item())
            summary[f"max_{key}"] = float(tensor_vals.max().item())
            summary[f"min_{key}"] = float(tensor_vals.min().item())
            summary[f"p10_{key}"] = float(torch.quantile(tensor_vals, 0.1).item())
            summary[f"p90_{key}"] = float(torch.quantile(tensor_vals, 0.9).item())
    quality_rows = [row for row in rows if row.get("prior_cue_quality_pass") is not None]
    if quality_rows:
        summary["cue_quality_pass_fraction"] = float(
            sum(1 for row in quality_rows if bool(row.get("prior_cue_quality_pass"))) / max(len(quality_rows), 1)
        )
    mass_rows = [row for row in rows if row.get("prior_dynamic_mass_D_gt_050") is not None]
    if mass_rows:
        summary["chunk_coverage_mass_gt_001"] = float(
            sum(1 for row in mass_rows if float(row.get("prior_dynamic_mass_D_gt_050") or 0.0) > 0.01)
            / max(len(mass_rows), 1)
        )
    with open(out_dir / "cue_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _write_hmc_correctness_summary(args: argparse.Namespace) -> None:
    out_dir = _infer_run_dir(args)
    if out_dir is None:
        return
    path = Path(args.hybrid_debug_jsonl) if args.hybrid_debug_jsonl else out_dir / "hmc_state_hash.jsonl"
    if not path.is_file():
        return
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return
    def _max_num(key: str) -> float:
        vals = [float(row.get(key, 0.0) or 0.0) for row in rows]
        return max(vals) if vals else 0.0

    hook_totals: Dict[str, int] = {}
    hook_max_bias: Dict[str, float] = {}
    for row in rows:
        trace = row.get("control_trace") or {}
        counts = trace.get("hook_trace_counts") or {}
        effects = trace.get("hook_effect_summary") or {}
        for key, value in counts.items():
            hook_totals[key] = hook_totals.get(key, 0) + int(value or 0)
        for key, value in effects.items():
            if isinstance(value, dict):
                hook_max_bias[key] = max(hook_max_bias.get(key, 0.0), float(value.get("max_abs_bias", 0.0) or 0.0))
    summary = {
        "num_chunks": len(rows),
        "probe_no_commit_hash_equal_all": all(bool(row.get("probe_no_commit_hash_equal")) for row in rows),
        "state_double_write_safe_all": all(bool(row.get("state_double_write_safe", True)) for row in rows),
        "max_pass1_pass2_pose_t_max": _max_num("pass1_pass2_pose_t_max"),
        "max_pass1_pass2_pose_matrix_abs_max": _max_num("pass1_pass2_pose_matrix_abs_max"),
        "hook_trace_call_totals": hook_totals,
        "hook_max_abs_bias": hook_max_bias,
    }
    with open(out_dir / "hmc_correctness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def _build_global_timestamps_for_output(image_paths: List[str], input_path: str) -> List[float]:
    timestamps = build_timestamps_for_output(image_paths, input_path)
    if image_paths and timestamps and abs(float(timestamps[0])) < 1e-9:
        parsed: List[float] = []
        ok = True
        for path in image_paths:
            try:
                parsed.append(float(int(Path(path).stem)))
            except ValueError:
                ok = False
                break
        if ok and parsed:
            timestamps = parsed
    return timestamps


def _save_outputs(
    *,
    args: argparse.Namespace,
    image_paths: List[str],
    images_loger: torch.Tensor,
    backbone: LoGeRGeometryBackbone,
    all_geo: List[GeometryOutput],
    window_raw_predictions: List[Dict[str, Any]],
    all_prior: List[Optional[PriorOutput]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if window_raw_predictions and args.chunk_size > 0:
        merged_raw = _merge_external_window_predictions(
            backbone,
            window_raw_predictions,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            reset_every=args.reset_every,
            se3=bool(args.se3),
        )
        T = images_loger.shape[0]
        H = images_loger.shape[2]
        W = images_loger.shape[3]
        patch_h, patch_w = H // 14, W // 14
        merged_geo = backbone._postprocess(
            merged_raw,
            images_loger.unsqueeze(0),
            T,
            H,
            W,
            patch_h,
            patch_w,
        )
        merged_local_points = merged_geo.local_points
        merged_world_points = merged_geo.world_points
        merged_camera_poses = merged_geo.camera_poses
        merged_confidence = merged_geo.confidence
    else:
        all_geo_for_merge = align_chunk_geometry_outputs(all_geo, args.chunk_overlap)
        merged_local_points = merge_chunk_tensor_tail_trim([g.local_points for g in all_geo_for_merge], args.chunk_overlap)
        merged_world_points = merge_chunk_tensor_tail_trim([g.world_points for g in all_geo_for_merge], args.chunk_overlap)
        merged_camera_poses = merge_chunk_tensor_tail_trim([g.camera_poses for g in all_geo_for_merge], args.chunk_overlap)
        merged_confidence = merge_chunk_tensor_tail_trim([g.confidence for g in all_geo_for_merge], args.chunk_overlap)

    if args.output_txt and merged_camera_poses is not None:
        timestamps = _build_global_timestamps_for_output(image_paths, args.input)
        twc = merged_camera_poses[:, :3, 3]
        qwc = mat_to_quat(merged_camera_poses[:, :3, :3])
        S = min(len(timestamps), twc.shape[0], qwc.shape[0])
        write_trajectory_txt(Path(args.output_txt), timestamps[:S], twc[:S].tolist(), qwc[:S].tolist())
        print(f"Saved trajectory to {args.output_txt}")

    if args.output_pt:
        os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
        save_dict: Dict[str, torch.Tensor] = {}
        if merged_local_points is not None:
            save_dict["local_points"] = merged_local_points
        if merged_world_points is not None:
            save_dict["points"] = merged_world_points
        if merged_camera_poses is not None:
            save_dict["camera_poses"] = merged_camera_poses
        if merged_confidence is not None:
            save_dict["conf"] = merged_confidence
        if all_prior and all_prior[0] is not None:
            save_dict["A_tok"] = all_prior[0].A_tok
            save_dict["A_pix"] = all_prior[0].A_pix
            save_dict["Elig_pix"] = all_prior[0].Elig_pix
            save_dict["A_mask"] = all_prior[0].A_mask
            save_dict["r_mask"] = all_prior[0].r_mask
            save_dict["B_chunk_geo"] = torch.tensor(all_prior[0].B_chunk_geo)
        torch.save(save_dict, args.output_pt)
        print(f"Saved output to {args.output_pt}")

    return merged_local_points, merged_world_points, merged_camera_poses, merged_confidence


def main() -> None:
    args = build_parser().parse_args()
    _apply_read_path_selector(args)
    mode = _resolve_hybrid_mode(args)
    _validate_args(args, mode)

    print("\nLoGeR Hybrid Memory Pipeline v2")
    print(f"  hybrid_memory_mode : {mode}")
    print("  read-path hooks    : real identity hook sites available")
    _write_hmc_config(args, mode)

    image_paths, temp_dir = collect_image_paths_geo(args.input, args.start_frame, args.end_frame, args.stride)
    if not image_paths:
        sys.exit("No images found.")
    total_frames = len(image_paths)
    print(f"Collected {total_frames} images.")

    needs_stage_b = (not args.geometry_eval_mode) and mode in {"ttt_write_only", "hybrid", "read_path_only", "probe_only"}
    needs_stage_c = needs_stage_b and args.stage_c_mode != "none"
    needs_prior = (not args.geometry_eval_mode) and mode in {"ttt_write_only", "hybrid"}

    images_full: Optional[torch.Tensor]
    if needs_stage_c:
        images_full = load_images_tensor(image_paths)
        print(f"Full-res images: {tuple(images_full.shape)}")
    else:
        images_full = None
        print("Full-res images: skipped")

    target_w, target_h = args.resolution if args.resolution else (None, None)
    images_loger = loger_load_images(image_paths, target_w=target_w, target_h=target_h)
    print(f"LoGeR-res images: {tuple(images_loger.shape)}")

    chunks = split_into_chunks(total_frames, args.chunk_size, args.chunk_overlap)
    print(f"\nChunk schedule: {len(chunks)} chunk(s), size={args.chunk_size or total_frames}, overlap={args.chunk_overlap}")
    for ci, (s, e) in enumerate(chunks):
        print(f"  chunk {ci}: frames [{s}, {e})")

    backbone_kwargs: Dict[str, Any] = dict(
        device=args.device,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
        edge_rtol=args.geometry_edge_rtol,
        update_ttt_weights=False,
    )
    if args.se3 is not None:
        backbone_kwargs["se3"] = args.se3

    print("\nLoading LoGeR model ...")
    t0 = time.time()
    backbone = LoGeRGeometryBackbone.from_config(args.checkpoint, args.config, **backbone_kwargs)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    extractor = _build_stage_b(args) if needs_stage_b else None
    build_kwargs, frontend_kwargs = _build_stage_c_kwargs(args) if needs_stage_c else ({}, {})
    prior_gen = (
        SemanticPriorGenerator(
            use_g_write_geo=bool(args.spg_use_g_write_geo),
            rho_sem=args.rho_sem,
            a_min_special=args.a_min_special,
            a_token_floor=args.a_token_floor,
        )
        if needs_prior else None
    )
    hmc = HybridMemoryController(
        device=args.device,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        eta_mean_preserve=(args.prior_policy == "eta_mean_preserving"),
        prior_branch_mask=args.prior_branch_mask,
        prior_layer_mode=args.prior_layer_mode,
        prior_single_layer=args.prior_single_layer,
        prior_layer_branch_policy=args.prior_layer_branch_policy,
        ttt_write_delta_scale=args.ttt_write_delta_scale,
        ttt_write_delta_scales=args.ttt_write_delta_scales,
        ttt_write_native_mix_scales=args.ttt_write_native_mix_scales,
        ttt_write_prior_transform_mode=args.ttt_write_prior_transform_mode,
        ttt_write_prior_anti_scale=args.ttt_write_prior_anti_scale,
        ttt_write_prior_gamma=args.ttt_write_prior_gamma,
        ttt_write_replay_feature_gate_mode=args.ttt_write_replay_feature_gate_mode,
        ttt_write_replay_feature_gate_rho=args.ttt_write_replay_feature_gate_rho,
        ttt_write_replay_feature_gate_min=args.ttt_write_replay_feature_gate_min,
        ttt_write_replay_feature_gate_branch_mask=args.ttt_write_replay_feature_gate_branch_mask,
        ttt_write_replay_token_filter_mode=args.ttt_write_replay_token_filter_mode,
        ttt_write_replay_token_filter_ratio=args.ttt_write_replay_token_filter_ratio,
        ttt_write_replay_token_filter_threshold=args.ttt_write_replay_token_filter_threshold,
        ttt_write_replay_token_filter_scope=args.ttt_write_replay_token_filter_scope,
        ttt_write_replay_token_filter_branch_mask=args.ttt_write_replay_token_filter_branch_mask,
        ttt_write_replay_token_filter_blend=args.ttt_write_replay_token_filter_blend,
        ttt_write_replay_token_filter_blend_mode=args.ttt_write_replay_token_filter_blend_mode,
        ttt_write_transient_delta_subtract_scale=args.ttt_write_transient_delta_subtract_scale,
        ttt_write_transient_delta_branch_mask=args.ttt_write_transient_delta_branch_mask,
        ttt_write_commit_ema_alpha=args.ttt_write_commit_ema_alpha,
        ttt_write_commit_ema_branch_mask=args.ttt_write_commit_ema_branch_mask,
        ttt_write_native_delta_gate_mode=args.ttt_write_native_delta_gate_mode,
        ttt_write_native_delta_gate_min_cos=args.ttt_write_native_delta_gate_min_cos,
        ttt_write_native_delta_gate_fallback=args.ttt_write_native_delta_gate_fallback,
        ttt_write_native_delta_gate_cap_ratio=args.ttt_write_native_delta_gate_cap_ratio,
        ttt_write_native_delta_gate_branch_mask=args.ttt_write_native_delta_gate_branch_mask,
        ttt_write_commit_filter_mode=args.ttt_write_commit_filter_mode,
        ttt_write_commit_filter_risk_source=args.ttt_write_commit_filter_risk_source,
        ttt_write_commit_filter_scope=args.ttt_write_commit_filter_scope,
        ttt_write_commit_filter_stat=args.ttt_write_commit_filter_stat,
        ttt_write_commit_filter_base=args.ttt_write_commit_filter_base,
        ttt_write_commit_filter_gain=args.ttt_write_commit_filter_gain,
        ttt_write_commit_filter_min=args.ttt_write_commit_filter_min,
        ttt_write_commit_filter_max=args.ttt_write_commit_filter_max,
        ttt_write_commit_filter_branch_mask=args.ttt_write_commit_filter_branch_mask,
        enable_frame_read_control=bool(args.enable_frame_read_control),
        enable_swa_read_control=bool(args.enable_swa_read_control),
        enable_ttt_apply_control=bool(args.enable_ttt_apply_control),
        enable_chunk_read_control=bool(args.enable_chunk_read_control),
        beta_frame=args.beta_frame,
        beta_swa=args.beta_swa,
        beta_chunk=args.beta_chunk,
        swa_gate_min=args.swa_gate_min,
        rho_ttt_apply=args.rho_ttt_apply,
        read_layer_mode=args.read_layer_mode,
        read_single_layer=args.read_single_layer,
        enable_swa_overlap_bias=bool(args.enable_swa_overlap_bias),
        swa_overlap_bias_beta=args.swa_overlap_bias_beta,
        swa_overlap_bias_min_keep=args.swa_overlap_bias_min_keep,
        swa_overlap_bias_mode=args.swa_overlap_bias_mode,
        swa_overlap_bias_layer_mode=args.swa_overlap_bias_layer_mode,
        swa_overlap_bias_single_layer=args.swa_overlap_bias_single_layer,
        enable_swa_overlap_source_gate=bool(args.enable_swa_overlap_source_gate),
        swa_overlap_source_gate_rho=args.swa_overlap_source_gate_rho,
        swa_overlap_source_gate_min=args.swa_overlap_source_gate_min,
        swa_overlap_source_gate_mode=args.swa_overlap_source_gate_mode,
        swa_overlap_source_gate_target=args.swa_overlap_source_gate_target,
        swa_overlap_source_gate_layer_mode=args.swa_overlap_source_gate_layer_mode,
        swa_overlap_source_gate_single_layer=args.swa_overlap_source_gate_single_layer,
        enable_swa_overlap_source_replace=bool(args.enable_swa_overlap_source_replace),
        swa_overlap_source_replace_alpha=args.swa_overlap_source_replace_alpha,
        swa_overlap_source_replace_mode=args.swa_overlap_source_replace_mode,
        swa_overlap_source_replace_target=args.swa_overlap_source_replace_target,
        swa_overlap_source_replace_layer_mode=args.swa_overlap_source_replace_layer_mode,
        swa_overlap_source_replace_single_layer=args.swa_overlap_source_replace_single_layer,
        read_cue_source=args.read_cue_source,
        read_topk_frac=args.read_topk_frac,
        frame_bias_mode=args.frame_bias_mode,
        chunk_bias_mode=args.chunk_bias_mode,
        swa_bias_mode=args.swa_bias_mode,
        flow_model=args.flow_model,
        flow_pair_stride=args.flow_pair_stride,
        flow_fb_thr=args.flow_fb_thr,
        flow_residual_thr=args.flow_residual_thr,
        gram_layer_groups=args.gram_layer_groups,
        read_calib_mode=args.read_calib_mode,
        read_target_mass=args.read_target_mass,
        read_calib_tau=args.read_calib_tau,
        read_blend_lambda=args.read_blend_lambda,
        read_quality_mass_min=args.read_quality_mass_min,
        read_quality_mass_max=args.read_quality_mass_max,
        read_quality_anchor_max=args.read_quality_anchor_max,
        read_quality_frag_max=args.read_quality_frag_max,
        beta_policy=args.beta_policy,
        beta_energy_target=args.beta_energy_target,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        read_protect_ref=bool(args.read_protect_ref),
        read_protect_static=bool(args.read_protect_static),
        read_protection_mode=args.read_protection_mode,
        read_ref_strength=args.read_ref_strength,
        read_overlap_frames=(args.chunk_overlap if int(args.read_overlap_frames) < 0 else args.read_overlap_frames),
        read_reset_frames=args.read_reset_frames,
        read_attention_q=args.read_attention_q,
        read_static_anchor_thr=args.read_static_anchor_thr,
        read_static_dyn_thr=args.read_static_dyn_thr,
        hmc_write_score_source=args.hmc_write_score_source,
        hmc_write_sparse_ratio=args.hmc_write_sparse_ratio,
        hmc_write_sparse_mode=args.hmc_write_sparse_mode,
        hmc_write_alpha=args.mp_alpha,
        hmc_write_min=args.mp_min,
        hmc_write_max=args.mp_max,
        ttt_apply_min_gate=args.ttt_apply_min_gate,
        enable_swa_write_control=bool(args.enable_swa_write_control),
        swa_write_mode=args.swa_write_mode,
        swa_write_rho=args.swa_write_rho,
        swa_write_min_gate=args.swa_write_min_gate,
        swa_write_sparse_ratio=args.swa_write_sparse_ratio,
        swa_write_layer_mode=args.swa_write_layer_mode,
        swa_write_single_layer=args.swa_write_single_layer,
        swa_write_scope=args.swa_write_scope,
        swa_write_keep_scope=args.swa_write_keep_scope,
        swa_write_score_source=args.swa_write_score_source,
        swa_write_cache_blend_alpha=args.swa_write_cache_blend_alpha,
        swa_write_cache_blend_mode=args.swa_write_cache_blend_mode,
        swa_write_cache_blend_target=args.swa_write_cache_blend_target,
        ttt_write_token_scope=args.ttt_write_token_scope,
        ttt_write_token_scope_floor=args.ttt_write_token_scope_floor,
        fast_cue_eval=bool(args.fast_cue_eval),
    )

    if args.load_hmc_state_at_chunk:
        state = _load_hmc_state(args.load_hmc_state_at_chunk)
        print(f"Loaded HMC state from {args.load_hmc_state_at_chunk}: {hybrid_state_fingerprint(state)}")
    else:
        state = HybridMemoryState()
    all_geo: List[GeometryOutput] = []
    all_cue: List[Optional[CueOutput]] = []
    all_masklet: List[Optional[MaskletOutput]] = []
    all_prior: List[Optional[PriorOutput]] = []
    window_raw_predictions: List[Dict[str, Any]] = []
    ttt_freeze_chunks = _parse_chunk_index_set(args.ttt_freeze_chunks)
    ttt_semantic_write_scale_chunks = _parse_chunk_float_map(args.ttt_semantic_write_scale_chunks)

    for ci, (start, end) in enumerate(chunks):
        print(f"\n{'#'*72}")
        print(f"# V2 Chunk {ci}/{len(chunks)-1} frames [{start}, {end})")
        print(f"{'#'*72}")

        if args.save_hmc_states:
            before_path = Path(args.save_hmc_states) / f"chunk_{ci:03d}_before.pt"
            _save_hmc_state(before_path, state)

        if args.stage_memory_swap and str(args.device).startswith("cuda"):
            _offload_stage_c_frontend_to_cpu()
            _ensure_backbone_on_device(backbone, args.device)

        if args.reset_every > 0 and ci > 0 and ci % args.reset_every == 0:
            state = _reset_hybrid_state_if_needed(state) or HybridMemoryState()
            print(f"  External reset_every triggered at chunk {ci}: cleared TTT fast weights, preserved local history")

        chunk_loger = images_loger[start:end]
        chunk_full = images_full[start:end] if images_full is not None else None

        print("\n  Pass 1: Probe Geometry Backbone ...")
        t0 = time.time()
        probe = hmc.run_probe(backbone, chunk_loger, state)
        probe_after_state_hash = hybrid_state_fingerprint(state)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    done in {time.time() - t0:.2f}s")
        print_geometry_output(probe.geometry)
        print(f"    committed state hash          : {probe.debug.get('committed_state_hash')}")
        print(f"    after-probe state hash        : {probe_after_state_hash}")
        print(f"    native provisional state hash : {probe.debug.get('native_provisional_state_hash')}")

        cue: Optional[CueOutput] = None
        if needs_stage_b:
            print("  Stage B: Dynamic / Internal Cue Extractor ...")
            t0 = time.time()
            assert extractor is not None
            cue = extractor.run(probe.geometry)
            print(f"    done in {time.time() - t0:.2f}s")
            print_cue_output(cue)
        else:
            print("  Stage B: skipped")
        all_cue.append(cue)

        mo: Optional[MaskletOutput] = None
        if needs_stage_b:
            if args.stage_c_mode == "none":
                print("  Stage C: skipped (--stage_c_mode=none)")
                H_p, W_p = probe.geometry.local_points.shape[1:3]
                mo = _empty_masklet_output(probe.geometry.num_frames, int(H_p), int(W_p))
            else:
                print("  Stage C: Video Masklet Front-end ...")
                t0 = time.time()
                assert chunk_full is not None
                if args.stage_memory_swap and str(args.device).startswith("cuda"):
                    _offload_backbone_to_cpu(backbone)
                mo = _run_stage_c_lazy(build_kwargs, frontend_kwargs, chunk_full, ci)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                print(f"    done in {time.time() - t0:.2f}s")
                print_masklet_output(mo)
                if args.stage_memory_swap and str(args.device).startswith("cuda"):
                    _offload_stage_c_frontend_to_cpu()
                    _ensure_backbone_on_device(backbone, args.device)
        else:
            print("  Stage C: skipped")
        all_masklet.append(mo)

        prior: Optional[PriorOutput] = None
        control_prior: Optional[HybridMemoryControlPrior] = None
        if needs_prior:
            print("  Stage D: Memory Control Prior Generator ...")
            t0 = time.time()
            assert prior_gen is not None and cue is not None and mo is not None
            prior = prior_gen.run(cue, mo, probe.geometry)
            prior = _apply_prior_policy(
                prior,
                cue,
                probe.geometry,
                policy=args.prior_policy,
                alpha=args.mp_alpha,
                p_min=args.mp_min,
                p_max=args.mp_max,
                score_source=args.mp_score_source,
            )
            prior = _apply_debug_prior_mode(
                prior,
                probe.geometry,
                mode=args.debug_prior_mode,
                roll_tokens=args.debug_prior_roll_tokens,
            )
            print(f"    done in {time.time() - t0:.2f}s")
            print_prior_output(prior)
        else:
            print("  Stage D: skipped")
        all_prior.append(prior)

        control_prior = hmc.build_control_prior(
            probe=probe,
            cue=cue,
            prior_output=prior,
            mode=mode,
        )

        result: Optional[HybridMemoryResult] = None
        if mode == "probe_only":
            print("  Pass 2: skipped (--hybrid_memory_mode=probe_only)")
            final_geo = probe.geometry
            state_next = probe.native_provisional_state
        else:
            print("  Pass 2: Controlled Geometry Backbone + Hybrid Memory Controller ...")
            t0 = time.time()
            result = hmc.run_controlled(
                backbone,
                chunk_loger,
                state,
                control_prior,
                mode=mode,
                token_type=probe.geometry.token_type,
                skip_ttt_write_replay=(args.hmc_commit_mode == "probe_ttt_write"),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"    done in {time.time() - t0:.2f}s")
            final_geo = result.geometry_output
            memory_side_effect = _state_side_effect_summary(
                controlled=result.state_next,
                probe_native=probe.native_provisional_state,
            )
            if args.hmc_commit_mode == "controlled":
                state_next = result.state_next
            elif args.hmc_commit_mode == "probe_native":
                state_next = probe.native_provisional_state
                if control_prior is not None:
                    state_next.prev_control_summary = hmc._build_prev_control_summary(control_prior, probe.geometry)
            elif args.hmc_commit_mode == "split_ttt_native":
                state_next = HybridMemoryState(
                    ttt_state=probe.native_provisional_state.ttt_state,
                    swa_state=result.state_next.swa_state,
                    ref_state=result.state_next.ref_state,
                    prev_control_summary=result.state_next.prev_control_summary,
                    debug=dict(result.state_next.debug),
                )
            elif args.hmc_commit_mode == "probe_ttt_write":
                state_next, probe_write_result = hmc.build_probe_ttt_write_state(
                    probe,
                    control_prior,
                    token_type=probe.geometry.token_type,
                    prev_ttt_state=state.ttt_state if state is not None else None,
                )
                result.debug["probe_ttt_write_state_hash"] = hybrid_state_fingerprint(state_next)
                result.debug["probe_ttt_write_debug"] = probe_write_result.debug if probe_write_result is not None else None
            else:
                raise ValueError(f"Unsupported --hmc_commit_mode={args.hmc_commit_mode}")
            if ci in ttt_semantic_write_scale_chunks and state_next is not None:
                scale = float(ttt_semantic_write_scale_chunks[ci])
                native_state = probe.native_provisional_state if probe is not None else None
                if native_state is not None and native_state.ttt_state is not None:
                    state_next = HybridMemoryState(
                        ttt_state=_blend_semantic_ttt_against_native(
                            native_state.ttt_state,
                            state_next.ttt_state,
                            scale,
                        ),
                        swa_state=state_next.swa_state,
                        ref_state=state_next.ref_state,
                        prev_control_summary=state_next.prev_control_summary,
                        debug={
                            **dict(state_next.debug),
                            "ttt_semantic_write_scale_at_chunk": {
                                "chunk": int(ci),
                                "scale": float(scale),
                                "base": "probe_native_provisional",
                            },
                        },
                    )
                    if isinstance(result.debug, dict):
                        result.debug["ttt_semantic_write_scale_at_chunk"] = {
                            "chunk": int(ci),
                            "scale": float(scale),
                            "base": "probe_native_provisional",
                        }
                        result.debug["ttt_semantic_scaled_state_hash"] = hybrid_state_fingerprint(state_next)
                    print(
                        "    diagnostic semantic TTT scale : "
                        f"chunk {ci}, semantic delta scale={scale:.4g}, native TTT preserved"
                    )
            if ci in ttt_freeze_chunks and state_next is not None:
                prev_ttt_state = state.ttt_state if state is not None else None
                state_next = HybridMemoryState(
                    ttt_state=prev_ttt_state,
                    swa_state=state_next.swa_state,
                    ref_state=state_next.ref_state,
                    prev_control_summary=state_next.prev_control_summary,
                    debug={**dict(state_next.debug), "ttt_write_commit_frozen_at_chunk": int(ci)},
                )
                if isinstance(result.debug, dict):
                    result.debug["ttt_write_commit_frozen_at_chunk"] = int(ci)
                    result.debug["frozen_ttt_state_hash"] = hybrid_state_fingerprint(state_next)
                print(f"    diagnostic TTT freeze        : chunk {ci}, discarded TTT write commit")
            print_geometry_output(final_geo)
            if result.write_result is not None:
                print_write_result(result.write_result)
            elif isinstance(result.debug.get("write_debug"), dict) and result.debug["write_debug"].get("semantic_write_skipped"):
                print("    controlled TTT write replay  : skipped")
                print(f"      reason                     : {result.debug['write_debug'].get('skip_reason')}")
            if args.hmc_commit_mode == "probe_ttt_write" and result.debug.get("probe_ttt_write_debug") is not None:
                print("    probe/native TTT write commit:")
                print(f"      state hash                  : {result.debug.get('probe_ttt_write_state_hash')}")
                print(f"      debug                       : {result.debug.get('probe_ttt_write_debug')}")
            print(f"    controlled input state hash  : {result.debug.get('controlled_input_state_hash')}")
            print(f"    controlled output state hash : {result.debug.get('controlled_output_state_hash')}")
            print(f"    commit mode                  : {args.hmc_commit_mode}")
            print(f"    committed output state hash  : {hybrid_state_fingerprint(state_next)}")
            print(
                "    memory side effect          : "
                f"ttt_mean_rel={memory_side_effect.get('ttt_state_diff', {}).get('mean_rel_diff', 0.0):.6g}, "
                f"ttt_max_rel={memory_side_effect.get('ttt_state_diff', {}).get('max_rel_diff', 0.0):.6g}"
            )
            print(f"    control paths                : {result.control_trace}")
        if mode == "probe_only":
            memory_side_effect: Optional[Dict[str, Any]] = None

        geometry_diff = _geometry_diff_summary(probe.geometry, final_geo)
        if geometry_diff:
            print(
                "    pass1/pass2 pose diff        : "
                f"t_mean={geometry_diff.get('pass1_pass2_pose_t_mean', 0.0):.6g}, "
                f"t_max={geometry_diff.get('pass1_pass2_pose_t_max', 0.0):.6g}, "
                f"r_mean={geometry_diff.get('pass1_pass2_pose_r_deg_mean', 0.0):.6g} deg"
            )

        _append_hybrid_debug_jsonl(
            args.hybrid_debug_jsonl,
            args=args,
            mode=mode,
            chunk_idx=ci,
            start=start,
            end=end,
            probe=probe,
            probe_after_state_hash=probe_after_state_hash,
            control_prior=control_prior,
            result=result,
            committed_state_next=state_next,
            memory_side_effect=memory_side_effect,
            geometry_diff=geometry_diff,
        )
        if args.prior_debug_jsonl and result is not None and result.write_result is not None:
            _append_prior_debug_jsonl(
                args.prior_debug_jsonl,
                args=args,
                chunk_idx=ci,
                start=start,
                end=end,
                geo=probe.geometry,
                prior=prior,
                wr=result.write_result,
            )

        all_geo.append(final_geo)
        window_raw_predictions.append(_rebuild_batched_raw_window(final_geo, start, end))

        state = state_next or HybridMemoryState()
        if args.stage_memory_swap and str(args.device).startswith("cuda") and state.ttt_state is not None:
            state.ttt_state = _move_tensor_tree_to_device(state.ttt_state, "cpu")
        print(f"  Committed H_next hash: {hybrid_state_fingerprint(state)}")
        if args.save_hmc_states:
            after_path = Path(args.save_hmc_states) / f"chunk_{ci:03d}_after.pt"
            _save_hmc_state(after_path, state)

        del probe, cue, mo, prior, control_prior, result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_hmc_correctness_summary(args)
    _write_cue_quality_summary(args)
    _save_outputs(
        args=args,
        image_paths=image_paths,
        images_loger=images_loger,
        backbone=backbone,
        all_geo=all_geo,
        window_raw_predictions=window_raw_predictions,
        all_prior=all_prior,
    )

    if all_masklet and all_cue and all_masklet[0] is not None and all_cue[0] is not None:
        mo0 = all_masklet[0]
        cue0 = all_cue[0]
        prior0 = all_prior[0] if all_prior else None
        dyn_scores = score_masklets_by_cdyn(mo0, cue0)
        if images_full is not None and (args.output_video or args.save_frames):
            create_video(
                images_full[:chunks[0][1]],
                mo0,
                args.output_video,
                fps=args.fps,
                alpha=args.mask_alpha,
                prior=prior0,
                per_masklet_dyn=dyn_scores,
                save_frames_dir=args.save_frames,
            )

    if temp_dir is not None:
        # collect_image_paths_geo owns cleanup in many callers, but keep this
        # script non-destructive: temporary extraction dirs are left for debug.
        pass


if __name__ == "__main__":
    main()
