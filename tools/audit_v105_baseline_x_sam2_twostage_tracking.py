#!/usr/bin/env python3
"""Baseline-x: SAM2 two-stage frame0 initialization plus SAM2 tracking.

This runner is intentionally separate from the earlier 4D_PM-style ablations.
Frame 0 uses a high-confidence 16x16 point grid that selects the largest valid
SAM2 candidate per point, then samples 200 points only in the still-uncovered
pixels and selects the smallest valid candidate. Later frames keep the 4D_PM
gap policy: propagate new masks with SAM2 video tracking, sample uncovered
regions, select smallest SAM2 masks, restrict them to uncovered pixels, and
propagate those new ids forward.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_sam2_twostage_sam2.generated.yaml"
STREAM_INFER_TRACE_HOOK = None

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v105_4dpm_largest_tracking_baseline import (  # noqa: E402
    clear_sam2_namespace,
    disjoin_keep_order,
    label_from_id_masks,
    make_numeric_frame_dir,
    make_sheet_grid,
    propagate_new_masks,
    sample_points_from_mask_yx,
    setup_models as setup_sam2_models,
    write_video,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    annotate_frame,
    disjoin_smallest_first,
    fix_mask_regions,
    make_points_yx_torch,
    mask_stats,
    overlay_label,
    parse_frame_ids,
    read_rgb,
    sha256_file,
    stable_seed,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid config: {path}")
    return data


def cfg_get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def make_args(config: dict[str, Any], cli: argparse.Namespace) -> SimpleNamespace:
    run_cfg = config.get("run", {})
    sam_cfg = config.get("sam2", {})
    s1 = config.get("frame0_stage1", {})
    s2 = config.get("frame0_stage2_uncovered", {})
    gap = config.get("gap", {})
    paths = config.get("paths", {})
    return SimpleNamespace(
        config_path=str(cli.config),
        baseline_id=str(cfg_get(config, "baseline", "id", default="baseline-x")),
        variant_id=str(cfg_get(config, "baseline", "variant", default="baseline_x_sam2_twostage_sam2")),
        model_provider=str(cfg_get(config, "baseline", "model_provider", default="sam2")),
        segmentor_name=str(cfg_get(config, "baseline", "segmentor", default="sam2.1_hiera_large")),
        tracker_name=str(cfg_get(config, "baseline", "tracker", default="sam2.1_hiera_large")),
        tracker_backend=str(cfg_get(config, "baseline", "tracker_backend", default="sam2")),
        tracker=str(cfg_get(config, "baseline", "tracker_backend", default="sam2")),
        scene_id=str(cli.scene_id or run_cfg.get("scene_id", "scene0011_00")),
        rgb_root=str(cli.rgb_root or run_cfg.get("rgb_root", "Stream3D/data/scannet/processed")),
        frame_start=int(cli.frame_start if cli.frame_start is not None else run_cfg.get("frame_start", 0)),
        frame_stride=int(cli.frame_stride if cli.frame_stride is not None else run_cfg.get("frame_stride", 5)),
        frame_count=int(cli.frame_count if cli.frame_count is not None else run_cfg.get("frame_count", 32)),
        frame_ids=str(cli.frame_ids if cli.frame_ids is not None else run_cfg.get("frame_ids", "")),
        output_root=str(cli.output_root or run_cfg.get("output_root", "Stream3D/outputs/audit/v105_baseline_x_sam2_twostage_scene0011_r1")),
        birth_dump_dir=str(getattr(cli, "birth_dump_dir", "") or run_cfg.get("birth_dump_dir", "")),
        seed=int(cli.seed if cli.seed is not None else run_cfg.get("seed", 105)),
        fps=float(run_cfg.get("fps", 8.0)),
        sheet_cell_width=int(run_cfg.get("sheet_cell_width", 520)),
        sam2_checkpoint=str(paths.get("sam2_checkpoint", "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt")),
        sam2_model_cfg=str(paths.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml")),
        efficienttam_root=str(paths.get("efficienttam_root", "third_party/EfficientTAM")),
        efficienttam_checkpoint=str(paths.get("efficienttam_checkpoint", "third_party/EfficientTAM/checkpoints/efficienttam_s.pt")),
        efficienttam_model_cfg=str(paths.get("efficienttam_model_cfg", "configs/efficienttam/efficienttam_s.yaml")),
        efficienttam_config_module=str(paths.get("efficienttam_config_module", "efficient_track_anything")),
        efficient_sam2_root=str(paths.get("efficient_sam2_root", "third_party/Efficient-SAM2")),
        efficient_sam2_checkpoint=str(paths.get("efficient_sam2_checkpoint", paths.get("sam2_checkpoint", "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"))),
        efficient_sam2_model_cfg=str(paths.get("efficient_sam2_model_cfg", paths.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"))),
        efficient_sam2_bypass_checkpoint=str(paths.get("efficient_sam2_bypass_checkpoint", "third_party/Efficient-SAM2/bypass/ckpt/bypass_bottleneck_large.pth")),
        efficient_sam2_model=str(paths.get("efficient_sam2_model", "large")),
        efficient_sam2_apply_wb=bool(paths.get("efficient_sam2_apply_wb", True)),
        efficient_sam2_apply_bypass=bool(paths.get("efficient_sam2_apply_bypass", True)),
        efficient_sam2_prune_memory=bool(paths.get("efficient_sam2_prune_memory", True)),
        efficient_sam2_topk_mask=bool(paths.get("efficient_sam2_topk_mask", True)),
        efficient_sam2_random_mask=bool(paths.get("efficient_sam2_random_mask", False)),
        efficient_sam2_uniform_mask=bool(paths.get("efficient_sam2_uniform_mask", False)),
        efficient_sam2_set_drop_ratio=float(paths.get("efficient_sam2_set_drop_ratio", 0.95)),
        efficient_sam2_mem_stride=int(paths.get("efficient_sam2_mem_stride", 1)),
        efficient_sam2_wb_theta=float(paths.get("efficient_sam2_wb_theta", 0.7)),
        efficient_sam2_mtp_theta=float(paths.get("efficient_sam2_mtp_theta", 0.5)),
        efficient_sam2_obj_theta=float(paths.get("efficient_sam2_obj_theta", 5.0)),
        efficient_sam2_dilate_mask=bool(paths.get("efficient_sam2_dilate_mask", True)),
        efficient_sam2_dilate_kernel_size=int(paths.get("efficient_sam2_dilate_kernel_size", 5)),
        efficient_sam2_bypass_type=str(paths.get("efficient_sam2_bypass_type", "bottleneck")),
        efficient_sam2_wb_all_layer=bool(paths.get("efficient_sam2_wb_all_layer", False)),
        efficient_sam2_disable_wb=bool(paths.get("efficient_sam2_disable_wb", False)),
        efficient_sam2_mem_filter=bool(paths.get("efficient_sam2_mem_filter", False)),
        efficient_sam2_small_bypass=bool(paths.get("efficient_sam2_small_bypass", False)),
        efficient_sam2_mem_frame_prune=bool(paths.get("efficient_sam2_mem_frame_prune", False)),
        efficient_sam2_num_frame_to_prune=int(paths.get("efficient_sam2_num_frame_to_prune", 0)),
        efficient_sam2_pool_memory=bool(paths.get("efficient_sam2_pool_memory", False)),
        efficient_sam2_pooling_ks=int(paths.get("efficient_sam2_pooling_ks", 2)),
        efficient_sam2_sel_max_iou=bool(paths.get("efficient_sam2_sel_max_iou", False)),
        points_per_batch=int(sam_cfg.get("points_per_batch", 128)),
        box_nms_thresh=float(sam_cfg.get("box_nms_thresh", 0.7)),
        stability_score_offset=float(sam_cfg.get("stability_score_offset", 1.0)),
        model_mask_thresh=float(sam_cfg.get("model_mask_thresh", 0.0)),
        empty_ratio=float(sam_cfg.get("empty_ratio", 0.001)),
        gap_inner_margin=int(sam_cfg.get("gap_inner_margin", 2)),
        offload_video_to_cpu=bool(sam_cfg.get("offload_video_to_cpu", True)),
        offload_state_to_cpu=bool(sam_cfg.get("offload_state_to_cpu", True)),
        propagation_chunk_size=int(sam_cfg.get("propagation_chunk_size", 0)),
        propagation_mode=str(sam_cfg.get("propagation_mode", "reseed_full_video")),
        model_dtype=str(sam_cfg.get("model_dtype", "float32")),
        runtime_num_maskmem=int(sam_cfg.get("runtime_num_maskmem", 0)),
        runtime_max_obj_ptrs_in_encoder=int(sam_cfg.get("runtime_max_obj_ptrs_in_encoder", 0)),
        runtime_max_cond_frames_in_attn=int(sam_cfg.get("runtime_max_cond_frames_in_attn", 0)),
        stream_keep_noncond_frames=int(sam_cfg.get("stream_keep_noncond_frames", 0)),
        stream_prune_invisible_after_frames=int(sam_cfg.get("stream_prune_invisible_after_frames", 0)),
        stream_prune_min_visible_area=int(sam_cfg.get("stream_prune_min_visible_area", 0)),
        stream_prune_protect_min_ever_area=int(sam_cfg.get("stream_prune_protect_min_ever_area", 0)),
        stream_prune_protect_max_objects=int(sam_cfg.get("stream_prune_protect_max_objects", 0)),
        stream_disjoin_claim_dropped=bool(sam_cfg.get("stream_disjoin_claim_dropped", True)),
        stream_disjoin_min_area_px=int(sam_cfg.get("stream_disjoin_min_area_px", 0)),
        stream_disjoin_recent_min_iou=float(sam_cfg.get("stream_disjoin_recent_min_iou", 0.0)),
        stream_disjoin_recent_max_area_growth=float(sam_cfg.get("stream_disjoin_recent_max_area_growth", 0.0)),
        stream_empty_cache_every=int(sam_cfg.get("stream_empty_cache_every", 0)),
        stream_empty_cache_on_prune=bool(sam_cfg.get("stream_empty_cache_on_prune", True)),
        stage1_point_mode=str(s1.get("point_mode", "4dpm_grid")),
        stage1_grid_side=int(s1.get("grid_side", 16)),
        stage1_num_pts=int(s1.get("num_pts", int(s1.get("grid_side", 16)) ** 2)),
        stage1_iou_threshold=float(s1.get("pred_iou_thresh", 0.8)),
        stage1_stability_threshold=float(s1.get("stability_score_thresh", 0.95)),
        stage1_choice_policy=str(s1.get("choice_policy", "largest_valid_mask_per_point")),
        stage1_apply_box_nms=bool(s1.get("apply_box_nms", True)),
        stage1_nms_score_type=str(s1.get("nms_score_type", "pred_iou")),
        stage2_num_pts=int(s2.get("num_pts", 200)),
        stage2_iou_threshold=float(s2.get("pred_iou_thresh", 0.8)),
        stage2_stability_threshold=float(s2.get("stability_score_thresh", 0.8)),
        stage2_choice_policy=str(s2.get("choice_policy", "smallest_valid_mask_per_point")),
        stage2_apply_box_nms=bool(s2.get("apply_box_nms", False)),
        gap_num_pts_active=int(gap.get("num_pts_active", 800)),
        gap_iou_threshold=float(gap.get("pred_iou_thresh", 0.8)),
        gap_stability_threshold=float(gap.get("stability_score_thresh", 0.8)),
        gap_choice_policy=str(gap.get("choice_policy", "smallest_valid_mask_per_point")),
        gap_apply_box_nms=bool(gap.get("apply_box_nms", False)),
        gap_sampler=str(gap.get("sampler", "uniform_random")),
        gap_max_points=int(gap.get("max_points", gap.get("num_pts_active", 800))),
        gap_min_component_area=int(gap.get("min_component_area", 400)),
        gap_base_points_per_component=int(gap.get("base_points_per_component", 1)),
        gap_area_per_extra_point=int(gap.get("area_per_extra_point", 40000)),
        gap_max_points_per_component=int(gap.get("max_points_per_component", 8)),
        gap_min_image_edge_distance_px=int(gap.get("min_image_edge_distance_px", 0)),
        gap_small_mask_max_area=int(gap.get("small_mask_max_area", 0)),
        gap_small_mask_min_pred_iou=float(gap.get("small_mask_min_pred_iou", 0.0)),
        gap_output_min_pred_iou=float(gap.get("output_min_pred_iou", 0.0)),
        gap_output_min_stability=float(gap.get("output_min_stability", 0.0)),
        gap_output_allow_relaxed=bool(gap.get("output_allow_relaxed", True)),
        gap_large_reuse_recent_id_iou=float(gap.get("large_reuse_recent_id_iou", 0.0)),
        gap_large_reuse_min_area=int(gap.get("large_reuse_min_area", 0)),
        gap_large_reuse_max_area_ratio=float(gap.get("large_reuse_max_area_ratio", 0.0)),
        gap_delayed_admission_enabled=bool(gap.get("delayed_admission_enabled", False)),
        gap_admission_min_pred_iou=float(gap.get("admission_min_pred_iou", 0.0)),
        gap_admission_min_stability=float(gap.get("admission_min_stability", 0.0)),
        gap_admission_allow_relaxed=bool(gap.get("admission_allow_relaxed", True)),
        gap_anti_merge_core_window_frames=int(gap.get("anti_merge_core_window_frames", 0)),
        gap_anti_merge_core_erode_px=int(gap.get("anti_merge_core_erode_px", 0)),
        gap_anti_merge_core_min_area=int(gap.get("anti_merge_core_min_area", 0)),
        gap_anti_merge_core_min_overlap_px=int(gap.get("anti_merge_core_min_overlap_px", 0)),
        gap_anti_merge_core_min_overlap_ratio=float(gap.get("anti_merge_core_min_overlap_ratio", 0.0)),
        gap_anti_merge_max_overlap_objects=int(gap.get("anti_merge_max_overlap_objects", 1)),
        output_large_reuse_recent_id_iou=float(gap.get("output_large_reuse_recent_id_iou", 0.0)),
        output_large_reuse_min_area=int(gap.get("output_large_reuse_min_area", 0)),
        output_large_reuse_max_area_ratio=float(gap.get("output_large_reuse_max_area_ratio", 0.0)),
        output_reuse_prevent_collision_union=bool(gap.get("output_reuse_prevent_collision_union", False)),
    )


def apply_runtime_model_tuning(model: Any, args: SimpleNamespace) -> dict[str, Any]:
    import torch

    dtype_name = str(args.model_dtype).lower()
    tuning: dict[str, Any] = {
        "requested_model_dtype": str(args.model_dtype),
        "requested_num_maskmem": int(args.runtime_num_maskmem),
        "requested_max_obj_ptrs_in_encoder": int(args.runtime_max_obj_ptrs_in_encoder),
        "requested_max_cond_frames_in_attn": int(getattr(args, "runtime_max_cond_frames_in_attn", 0)),
    }
    if dtype_name in {"bf16", "bfloat16"}:
        model.to(dtype=torch.bfloat16)
        tuning["applied_model_dtype"] = "bfloat16"
    elif dtype_name in {"fp16", "float16"}:
        model.to(dtype=torch.float16)
        tuning["applied_model_dtype"] = "float16"
    elif dtype_name in {"", "fp32", "float32", "none"}:
        tuning["applied_model_dtype"] = "float32"
    else:
        raise ValueError(f"unsupported sam2.model_dtype: {args.model_dtype}")
    if int(args.runtime_num_maskmem) > 0:
        model.num_maskmem = int(args.runtime_num_maskmem)
    if int(args.runtime_max_obj_ptrs_in_encoder) > 0:
        model.max_obj_ptrs_in_encoder = int(args.runtime_max_obj_ptrs_in_encoder)
    if int(getattr(args, "runtime_max_cond_frames_in_attn", 0)) > 0:
        model.max_cond_frames_in_attn = int(args.runtime_max_cond_frames_in_attn)
    tuning["effective_num_maskmem"] = int(getattr(model, "num_maskmem", -1))
    tuning["effective_max_obj_ptrs_in_encoder"] = int(getattr(model, "max_obj_ptrs_in_encoder", -1))
    tuning["effective_max_cond_frames_in_attn"] = int(getattr(model, "max_cond_frames_in_attn", -1))
    return tuning


def setup_models(args: SimpleNamespace) -> dict[str, Any]:
    provider = str(args.model_provider).lower()
    if provider == "sam2":
        repo_path = REPO_ROOT / "Grounded-SAM-2"
        clear_sam2_namespace(repo_path)
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_checkpoint = (REPO_ROOT / args.sam2_checkpoint).resolve() if not Path(args.sam2_checkpoint).is_absolute() else Path(args.sam2_checkpoint)
        sam2_cfg = str(args.sam2_model_cfg)
        image_model = build_sam2(sam2_cfg, str(sam2_checkpoint), device="cuda")
        segmentor = SAM2ImagePredictor(image_model)
        tracker_model = build_sam2_video_predictor(sam2_cfg, str(sam2_checkpoint), device="cuda")
        segmentor_tuning = apply_runtime_model_tuning(image_model, args)
        tracker_tuning = apply_runtime_model_tuning(tracker_model, args)
        return {
            "model_provider": "sam2",
            "segmentor": segmentor,
            "tracker_model": tracker_model,
            "segmentor_name": str(args.segmentor_name),
            "tracker_name": str(args.tracker_name),
            "segmentor_checkpoint": sam2_checkpoint,
            "segmentor_cfg": sam2_cfg,
            "tracker_checkpoint": sam2_checkpoint,
            "tracker_cfg": sam2_cfg,
            "sam2_checkpoint": sam2_checkpoint,
            "sam2_cfg": sam2_cfg,
            "runtime_tuning": {
                "segmentor": segmentor_tuning,
                "tracker": tracker_tuning,
            },
        }
    if provider == "efficient_sam2":
        import torch

        root = (REPO_ROOT / args.efficient_sam2_root).resolve() if not Path(args.efficient_sam2_root).is_absolute() else Path(args.efficient_sam2_root)
        clear_sam2_namespace(root)

        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from build_model.window_bypass import build_WB_model
        from bypass.bypass_modeling import build_bypass_model

        checkpoint = (REPO_ROOT / args.efficient_sam2_checkpoint).resolve() if not Path(args.efficient_sam2_checkpoint).is_absolute() else Path(args.efficient_sam2_checkpoint)
        bypass_checkpoint = (REPO_ROOT / args.efficient_sam2_bypass_checkpoint).resolve() if not Path(args.efficient_sam2_bypass_checkpoint).is_absolute() else Path(args.efficient_sam2_bypass_checkpoint)
        cfg = str(args.efficient_sam2_model_cfg)
        image_model = build_sam2(cfg, str(checkpoint), device="cuda")
        segmentor = SAM2ImagePredictor(image_model)
        tracker_model = build_sam2_video_predictor(cfg, str(checkpoint), device="cuda")

        sam2_model = str(args.efficient_sam2_model)
        if sam2_model == "base+":
            if bool(args.efficient_sam2_wb_all_layer):
                selected_layers = list(range(0, 21))
                win_sel_layer = [0, 3, 6]
                fpn_feat_layer = [1, 4]
                scale_layer = [2, 5]
            else:
                selected_layers = list(range(6, 21))
                win_sel_layer = [6]
                fpn_feat_layer = None
                scale_layer = None
            final_global_layer = 20
            inner_channel = 448
        else:
            sam2_model = "large"
            if bool(args.efficient_sam2_wb_all_layer):
                selected_layers = list(range(0, 44))
                win_sel_layer = [0, 3, 9]
                fpn_feat_layer = [1, 7]
                scale_layer = [2, 8]
            else:
                selected_layers = list(range(9, 44))
                win_sel_layer = [9]
                fpn_feat_layer = None
                scale_layer = None
            final_global_layer = 43
            inner_channel = 576

        es_args = SimpleNamespace(
            apply_WB=bool(args.efficient_sam2_apply_wb),
            prune_memory=bool(args.efficient_sam2_prune_memory),
            apply_bypass=bool(args.efficient_sam2_apply_bypass),
            disable_WB=bool(args.efficient_sam2_disable_wb),
            Mem_filter=bool(args.efficient_sam2_mem_filter),
            small_bypass=bool(args.efficient_sam2_small_bypass),
            WB_all_layer=bool(args.efficient_sam2_wb_all_layer),
            pool_memory=bool(args.efficient_sam2_pool_memory),
            pooling_ks=int(args.efficient_sam2_pooling_ks),
            sam2_model=sam2_model,
            selected_layers=selected_layers,
            win_sel_layer=win_sel_layer,
            fpn_feat_layer=fpn_feat_layer,
            scale_layer=scale_layer,
            final_global_layer=final_global_layer,
            inner_channel=inner_channel,
            bypass_type=str(args.efficient_sam2_bypass_type),
            bypass_ckpt=str(bypass_checkpoint),
            topk_mask=bool(args.efficient_sam2_topk_mask),
            random_mask=bool(args.efficient_sam2_random_mask),
            uniform_mask=bool(args.efficient_sam2_uniform_mask),
            set_drop_ratio=float(args.efficient_sam2_set_drop_ratio),
            MTP_theta=float(args.efficient_sam2_mtp_theta),
            Mem_Frame_Prune=bool(args.efficient_sam2_mem_frame_prune),
            num_frame_to_prune=int(args.efficient_sam2_num_frame_to_prune) if bool(args.efficient_sam2_mem_frame_prune) else 0,
            WB_theta=float(args.efficient_sam2_wb_theta),
            obj_theta=float(args.efficient_sam2_obj_theta),
            sel_max_iou=bool(args.efficient_sam2_sel_max_iou),
        )

        tracker_model.memory_temporal_stride_for_eval = int(args.efficient_sam2_mem_stride)
        tracker_model.print_WS = False
        tracker_model.random_mask = bool(args.efficient_sam2_random_mask)
        tracker_model.uniform_mask = bool(args.efficient_sam2_uniform_mask)
        tracker_model.topk_mask = bool(args.efficient_sam2_topk_mask)
        tracker_model.set_drop_ratio = float(args.efficient_sam2_set_drop_ratio)
        tracker_model.MTP_theta = float(args.efficient_sam2_mtp_theta)
        tracker_model.disable_WB = bool(args.efficient_sam2_disable_wb)
        tracker_model.sam2_model = sam2_model
        tracker_model.win_sel_layer = win_sel_layer
        tracker_model.fpn_feat_layer = fpn_feat_layer
        tracker_model.scale_layer = scale_layer
        tracker_model.Mem_Frame_Prune = bool(args.efficient_sam2_mem_frame_prune)
        tracker_model.num_frame_to_prune = es_args.num_frame_to_prune
        tracker_model.WB_theta = float(args.efficient_sam2_wb_theta)
        tracker_model.obj_theta = float(args.efficient_sam2_obj_theta)
        tracker_model.sel_max_iou = bool(args.efficient_sam2_sel_max_iou)
        tracker_model.init_memory_info(enable_MeP_info=bool(args.efficient_sam2_prune_memory))
        if bool(args.efficient_sam2_apply_wb):
            build_WB_model(es_args, tracker_model, selected_layers=selected_layers)
            if bool(args.efficient_sam2_apply_bypass):
                build_bypass_model(es_args, tracker_model, adapter_type=str(args.efficient_sam2_bypass_type), adapter=True, training=False)
                tracker_model.image_encoder.trunk.blocks[final_global_layer].bypass_branch.load_state_dict(torch.load(str(bypass_checkpoint)))
                tracker_model.image_encoder.trunk.blocks[final_global_layer].bypass_branch.eval()
            else:
                build_bypass_model(es_args, tracker_model, adapter=False, training=False)
        tracker_model.dilate_mask = bool(args.efficient_sam2_dilate_mask)
        tracker_model.dilate_kernel_size = int(args.efficient_sam2_dilate_kernel_size)
        return {
            "model_provider": "efficient_sam2",
            "segmentor": segmentor,
            "tracker_model": tracker_model,
            "segmentor_name": str(args.segmentor_name),
            "tracker_name": str(args.tracker_name),
            "segmentor_checkpoint": checkpoint,
            "segmentor_cfg": cfg,
            "tracker_checkpoint": checkpoint,
            "tracker_cfg": cfg,
            "sam2_checkpoint": checkpoint,
            "sam2_cfg": cfg,
            "efficient_sam2_root": root,
            "efficient_sam2_bypass_checkpoint": bypass_checkpoint,
            "efficient_sam2_runtime": {
                "sam2_model": sam2_model,
                "apply_WB": bool(args.efficient_sam2_apply_wb),
                "apply_bypass": bool(args.efficient_sam2_apply_bypass),
                "prune_memory": bool(args.efficient_sam2_prune_memory),
                "topk_mask": bool(args.efficient_sam2_topk_mask),
                "set_drop_ratio": float(args.efficient_sam2_set_drop_ratio),
                "Mem_stride": int(args.efficient_sam2_mem_stride),
                "WB_theta": float(args.efficient_sam2_wb_theta),
                "dilate_mask": bool(args.efficient_sam2_dilate_mask),
                "dilate_kernel_size": int(args.efficient_sam2_dilate_kernel_size),
                "selected_layers": selected_layers,
                "win_sel_layer": win_sel_layer,
                "final_global_layer": int(final_global_layer),
                "inner_channel": int(inner_channel),
                "bypass_type": str(args.efficient_sam2_bypass_type),
                "tracker_class": type(tracker_model).__name__,
                "trunk_class": type(tracker_model.image_encoder.trunk).__name__,
            },
        }
    if provider != "efficienttam":
        raise ValueError(f"unsupported model_provider={args.model_provider}")

    from hydra import initialize_config_module
    from hydra.core.global_hydra import GlobalHydra

    root = (REPO_ROOT / args.efficienttam_root).resolve() if not Path(args.efficienttam_root).is_absolute() else Path(args.efficienttam_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    initialize_config_module(config_module=str(args.efficienttam_config_module), version_base="1.3.2")

    from efficient_track_anything.build_efficienttam import build_efficienttam, build_efficienttam_video_predictor
    from efficient_track_anything.efficienttam_image_predictor import EfficientTAMImagePredictor

    checkpoint = (REPO_ROOT / args.efficienttam_checkpoint).resolve() if not Path(args.efficienttam_checkpoint).is_absolute() else Path(args.efficienttam_checkpoint)
    cfg = str(args.efficienttam_model_cfg)
    image_model = build_efficienttam(cfg, str(checkpoint), device="cuda")
    segmentor = EfficientTAMImagePredictor(image_model)
    tracker_model = build_efficienttam_video_predictor(cfg, str(checkpoint), device="cuda")
    return {
        "model_provider": "efficienttam",
        "segmentor": segmentor,
        "tracker_model": tracker_model,
        "segmentor_name": str(args.segmentor_name),
        "tracker_name": str(args.tracker_name),
        "segmentor_checkpoint": checkpoint,
        "segmentor_cfg": cfg,
        "tracker_checkpoint": checkpoint,
        "tracker_cfg": cfg,
        "sam2_checkpoint": checkpoint,
        "sam2_cfg": cfg,
    }


def disjoin_smallest_first_with_records(
    masks: np.ndarray,
    records: list[dict[str, Any]],
    h: int,
    w: int,
    *,
    empty_ratio: float,
    fix_small_regions: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if masks.size == 0:
        return np.zeros((0, h, w), dtype=bool), []
    if masks.ndim == 2:
        masks = masks[None, :, :]
    areas = np.count_nonzero(masks.reshape(masks.shape[0], -1), axis=1)
    rows: list[np.ndarray] = []
    row_records: list[dict[str, Any]] = []
    for idx in np.argsort(areas):
        idx_i = int(idx)
        mask = masks[idx_i].astype(bool)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if fix_small_regions:
            mask = fix_mask_regions(mask, area_thresh=64)
        record = dict(records[idx_i]) if idx_i < len(records) else {}
        record["pre_disjoint_rank"] = int(idx_i)
        rows.append(mask)
        row_records.append(record)

    claimed = np.zeros((h, w), dtype=bool)
    kept: list[np.ndarray] = []
    kept_records: list[dict[str, Any]] = []
    min_pixels = int(h * w * float(empty_ratio))
    for sorted_rank, (mask, record) in enumerate(zip(rows, row_records, strict=False)):
        residual = mask & ~claimed
        residual_area = int(np.count_nonzero(residual))
        if residual_area > min_pixels:
            out_record = dict(record)
            out_record["post_disjoint_rank"] = int(len(kept_records))
            out_record["sorted_area_rank"] = int(sorted_rank)
            out_record["post_disjoint_area_px"] = int(residual_area)
            kept.append(residual)
            kept_records.append(out_record)
        claimed |= mask
    if kept:
        return np.stack(kept, axis=0).astype(bool), kept_records
    return np.zeros((0, h, w), dtype=bool), []


def mask_shape_features(mask: np.ndarray) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    h, w = mask_b.shape[:2]
    image_area = max(1, int(h) * int(w))
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
            "area_px": 0,
            "bbox_xyxy": [0, 0, -1, -1],
            "bbox_area_frac": 0.0,
            "edge_touch_count": 0,
            "extent": 0.0,
            "core16_area_px": 0,
        }
    ys, xs = np.where(mask_b)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    edge_touch_count = int(x0 == 0) + int(x1 == w - 1) + int(y0 == 0) + int(y1 == h - 1)
    dist = cv2.distanceTransform(mask_b.astype(np.uint8), cv2.DIST_L2, 3)
    return {
        "area_px": int(area),
        "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_area_frac": float((bw * bh) / image_area),
        "edge_touch_count": int(edge_touch_count),
        "extent": float(area / max(1, bw * bh)),
        "core16_area_px": int(np.count_nonzero(dist >= 16.0)),
    }


def build_recent_ownership_cores(
    ownership_mask_by_obj: dict[int, np.ndarray] | None,
    ownership_frame_by_obj: dict[int, int] | None,
    *,
    current_frame_idx: int,
    window_frames: int,
    min_area: int,
    erode_px: int,
) -> list[dict[str, Any]]:
    if not ownership_mask_by_obj or not ownership_frame_by_obj:
        return []
    window = max(0, int(window_frames))
    if window <= 0:
        return []
    min_area_i = max(0, int(min_area))
    erode_i = max(0, int(erode_px))
    kernel = None
    if erode_i > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_i + 1, 2 * erode_i + 1))
    cores: list[dict[str, Any]] = []
    for obj_id_raw, mask_raw in ownership_mask_by_obj.items():
        obj_id = int(obj_id_raw)
        last_frame = int(ownership_frame_by_obj.get(obj_id, -10**9))
        age = int(current_frame_idx) - last_frame
        if age < 0 or age > window:
            continue
        mask = np.asarray(mask_raw).astype(bool, copy=False)
        area = int(np.count_nonzero(mask))
        if area < min_area_i:
            continue
        core = mask
        if kernel is not None:
            core = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        core_area = int(np.count_nonzero(core))
        if core_area <= 0:
            continue
        cores.append(
            {
                "object_id": int(obj_id),
                "last_frame_idx": int(last_frame),
                "age_frames": int(age),
                "area_px": int(area),
                "core_area_px": int(core_area),
                "core_mask": core,
            }
        )
    return cores


def find_multi_core_overlaps(
    mask: np.ndarray,
    *,
    ownership_cores: list[dict[str, Any]],
    min_overlap_px: int,
    min_overlap_ratio: float,
) -> list[dict[str, Any]]:
    if not ownership_cores:
        return []
    mask_b = np.asarray(mask).astype(bool, copy=False)
    min_overlap_px_i = max(0, int(min_overlap_px))
    min_overlap_ratio_f = max(0.0, float(min_overlap_ratio))
    matches: list[dict[str, Any]] = []
    for core_record in ownership_cores:
        core = np.asarray(core_record.get("core_mask")).astype(bool, copy=False)
        core_area = int(core_record.get("core_area_px", int(np.count_nonzero(core))))
        if core_area <= 0:
            continue
        inter = int(np.count_nonzero(mask_b & core))
        if inter <= 0:
            continue
        ratio = float(inter) / float(max(core_area, 1))
        if inter < min_overlap_px_i:
            continue
        if min_overlap_ratio_f > 0.0 and ratio < min_overlap_ratio_f:
            continue
        matches.append(
            {
                "object_id": int(core_record["object_id"]),
                "last_frame_idx": int(core_record["last_frame_idx"]),
                "age_frames": int(core_record["age_frames"]),
                "area_px": int(core_record["area_px"]),
                "core_area_px": int(core_area),
                "overlap_px": int(inter),
                "overlap_core_ratio": float(ratio),
            }
        )
    matches.sort(key=lambda row: (float(row["overlap_core_ratio"]), int(row["overlap_px"])), reverse=True)
    return matches


def filter_gap_output_masks_by_quality(
    masks: np.ndarray,
    records: list[dict[str, Any]],
    *,
    min_pred_iou: float,
    min_stability: float,
    allow_relaxed: bool,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    input_count = int(masks.shape[0]) if masks.size else 0
    min_pred_iou_f = max(0.0, float(min_pred_iou))
    min_stability_f = max(0.0, float(min_stability))
    enabled = bool(min_pred_iou_f > 0.0 or min_stability_f > 0.0 or not bool(allow_relaxed))
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "input_mask_count": int(input_count),
        "output_mask_count": int(input_count),
        "dropped_mask_count": 0,
        "min_pred_iou": float(min_pred_iou_f),
        "min_stability": float(min_stability_f),
        "allow_relaxed": bool(allow_relaxed),
        "dropped_mask_records_sample": [],
    }
    if input_count <= 0 or not enabled:
        return masks, records, stats

    keep = np.ones((input_count,), dtype=bool)
    dropped_records: list[dict[str, Any]] = []
    for idx, mask in enumerate(masks.astype(bool)):
        idx_i = int(idx)
        record = records[idx_i] if idx_i < len(records) else {}
        pred_iou = float(record.get("chosen_pred_iou", -1.0))
        stability = float(record.get("chosen_stability", -1.0))
        relaxed_selected = bool(record.get("relaxed_selected", False))
        reasons: list[str] = []
        if min_pred_iou_f > 0.0:
            if pred_iou < 0.0:
                reasons.append("missing_pred_iou_record")
            elif pred_iou < min_pred_iou_f:
                reasons.append("pred_iou_below_min")
        if min_stability_f > 0.0:
            if stability < 0.0:
                reasons.append("missing_stability_record")
            elif stability < min_stability_f:
                reasons.append("stability_below_min")
        if not bool(allow_relaxed) and relaxed_selected:
            reasons.append("relaxed_candidate_not_output")
        if reasons:
            keep[idx_i] = False
            dropped_records.append(
                {
                    "rank": int(idx_i),
                    "area_px": int(np.count_nonzero(mask)),
                    "chosen_pred_iou": float(pred_iou),
                    "chosen_stability": float(stability),
                    "relaxed_selected": bool(relaxed_selected),
                    "reasons": reasons,
                }
            )

    stats["output_mask_count"] = int(np.count_nonzero(keep))
    stats["dropped_mask_count"] = int(input_count - int(np.count_nonzero(keep)))
    stats["dropped_mask_records_sample"] = dropped_records[:64]
    if bool(np.all(keep)):
        return masks, records, stats
    filtered_masks = masks[keep] if masks.size else np.zeros_like(masks)
    filtered_records = [dict(records[i]) for i in range(min(len(records), input_count)) if bool(keep[i])]
    return filtered_masks, filtered_records, stats


def select_gap_admission_masks(
    obj_ids: np.ndarray,
    masks: np.ndarray,
    *,
    gap_records: list[dict[str, Any]],
    uncovered_mask: np.ndarray,
    ownership_mask_by_obj: dict[int, np.ndarray] | None = None,
    ownership_frame_by_obj: dict[int, int] | None = None,
    enabled: bool,
    frame_idx: int,
    frame_id: int,
    min_area: int,
    max_area: int,
    max_uncovered_ratio: float,
    max_bbox_frac: float,
    max_edge_touch_count: int,
    min_extent: float,
    min_core_area: int,
    shape_min_uncovered_ratio: float,
    max_per_frame: int,
    min_pred_iou: float,
    min_stability: float,
    allow_relaxed: bool,
    anti_merge_core_window_frames: int = 0,
    anti_merge_core_erode_px: int = 0,
    anti_merge_core_min_area: int = 0,
    anti_merge_core_min_overlap_px: int = 0,
    anti_merge_core_min_overlap_ratio: float = 0.0,
    anti_merge_max_overlap_objects: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    input_count = int(masks.shape[0]) if masks.size else 0
    uncovered_bool = np.asarray(uncovered_mask).astype(bool)
    uncovered_area = int(np.count_nonzero(uncovered_bool))
    image_area = int(uncovered_bool.size)
    current_uncovered_ratio = float(uncovered_area) / float(max(image_area, 1))
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "frame_idx": int(frame_idx),
        "frame_id": int(frame_id),
        "input_mask_count": int(input_count),
        "admitted_mask_count": int(input_count),
        "output_only_mask_count": 0,
        "current_uncovered_area_px": int(uncovered_area),
        "current_uncovered_ratio": float(current_uncovered_ratio),
        "min_area": int(max(0, int(min_area))),
        "max_area": int(max(0, int(max_area))),
        "max_uncovered_ratio": float(max(0.0, float(max_uncovered_ratio))),
        "max_bbox_frac": float(max(0.0, float(max_bbox_frac))),
        "max_edge_touch_count": int(max_edge_touch_count),
        "min_extent": float(max(0.0, float(min_extent))),
        "min_core_area": int(max(0, int(min_core_area))),
        "shape_min_uncovered_ratio": float(max(0.0, float(shape_min_uncovered_ratio))),
        "max_per_frame": int(max(0, int(max_per_frame))),
        "min_pred_iou": float(max(0.0, float(min_pred_iou))),
        "min_stability": float(max(0.0, float(min_stability))),
        "allow_relaxed": bool(allow_relaxed),
        "anti_merge_core_window_frames": int(max(0, int(anti_merge_core_window_frames))),
        "anti_merge_core_erode_px": int(max(0, int(anti_merge_core_erode_px))),
        "anti_merge_core_min_area": int(max(0, int(anti_merge_core_min_area))),
        "anti_merge_core_min_overlap_px": int(max(0, int(anti_merge_core_min_overlap_px))),
        "anti_merge_core_min_overlap_ratio": float(max(0.0, float(anti_merge_core_min_overlap_ratio))),
        "anti_merge_max_overlap_objects": int(max(1, int(anti_merge_max_overlap_objects))),
        "anti_merge_core_gate_enabled": False,
        "anti_merge_core_count": 0,
        "anti_merge_rejected_count": 0,
        "shape_filter_enabled": bool(
            float(max_bbox_frac) > 0.0
            or int(max_edge_touch_count) >= 0
            or float(min_extent) > 0.0
            or int(min_core_area) > 0
        ),
        "shape_filter_active": False,
        "admitted_obj_ids": [int(v) for v in obj_ids.tolist()] if obj_ids.size else [],
        "output_only_obj_ids": [],
        "output_only_records_sample": [],
    }
    if input_count <= 0:
        return obj_ids, masks, stats
    if not enabled:
        return obj_ids, masks, stats

    min_area_i = max(0, int(min_area))
    max_area_i = max(0, int(max_area))
    max_uncovered_ratio_f = max(0.0, float(max_uncovered_ratio))
    max_bbox_frac_f = max(0.0, float(max_bbox_frac))
    max_edge_touch_i = int(max_edge_touch_count)
    min_extent_f = max(0.0, float(min_extent))
    min_core_area_i = max(0, int(min_core_area))
    shape_min_uncovered_ratio_f = max(0.0, float(shape_min_uncovered_ratio))
    min_pred_iou_f = max(0.0, float(min_pred_iou))
    min_stability_f = max(0.0, float(min_stability))
    anti_merge_window_i = max(0, int(anti_merge_core_window_frames))
    anti_merge_erode_i = max(0, int(anti_merge_core_erode_px))
    anti_merge_min_area_i = max(0, int(anti_merge_core_min_area))
    anti_merge_min_overlap_px_i = max(0, int(anti_merge_core_min_overlap_px))
    anti_merge_min_overlap_ratio_f = max(0.0, float(anti_merge_core_min_overlap_ratio))
    anti_merge_max_objects_i = max(1, int(anti_merge_max_overlap_objects))
    anti_merge_enabled = bool(
        anti_merge_window_i > 0
        and anti_merge_min_area_i > 0
        and (anti_merge_min_overlap_px_i > 0 or anti_merge_min_overlap_ratio_f > 0.0)
    )
    shape_filter_enabled = bool(
        max_bbox_frac_f > 0.0
        or max_edge_touch_i >= 0
        or min_extent_f > 0.0
        or min_core_area_i > 0
    )
    shape_filter_active = bool(shape_filter_enabled and current_uncovered_ratio >= shape_min_uncovered_ratio_f)
    stats["shape_filter_active"] = bool(shape_filter_active)
    ownership_cores = build_recent_ownership_cores(
        ownership_mask_by_obj,
        ownership_frame_by_obj,
        current_frame_idx=int(frame_idx),
        window_frames=int(anti_merge_window_i),
        min_area=int(anti_merge_min_area_i),
        erode_px=int(anti_merge_erode_i),
    ) if anti_merge_enabled else []
    stats["anti_merge_core_gate_enabled"] = bool(anti_merge_enabled)
    stats["anti_merge_core_count"] = int(len(ownership_cores))

    areas = np.asarray([int(np.count_nonzero(mask)) for mask in masks.astype(bool)], dtype=np.int64)
    keep = np.ones((input_count,), dtype=bool)
    output_only_records: list[dict[str, Any]] = []
    anti_merge_rejected_count = 0
    admission_scores: list[tuple[float, float, float, int]] = []
    for idx, (obj_id_raw, mask) in enumerate(zip(obj_ids.tolist(), masks.astype(bool), strict=False)):
        idx_i = int(idx)
        obj_id = int(obj_id_raw)
        area = int(areas[idx_i])
        features = mask_shape_features(mask)
        record = gap_records[idx_i] if idx_i < len(gap_records) else {}
        pred_iou = float(record.get("chosen_pred_iou", -1.0))
        stability = float(record.get("chosen_stability", -1.0))
        relaxed_selected = bool(record.get("relaxed_selected", False))
        ratio = float(area) / float(max(uncovered_area, 1))
        reasons: list[str] = []
        if min_area_i > 0 and area < min_area_i:
            reasons.append("area_below_min")
        if max_area_i > 0 and area > max_area_i:
            reasons.append("area_above_max")
        if max_uncovered_ratio_f > 0.0 and ratio > max_uncovered_ratio_f:
            reasons.append("mask_to_uncovered_ratio_above_max")
        if shape_filter_active and max_bbox_frac_f > 0.0 and float(features["bbox_area_frac"]) > max_bbox_frac_f:
            reasons.append("bbox_frac_above_max")
        if shape_filter_active and max_edge_touch_i >= 0 and int(features["edge_touch_count"]) > max_edge_touch_i:
            reasons.append("edge_touch_above_max")
        if shape_filter_active and min_extent_f > 0.0 and float(features["extent"]) < min_extent_f:
            reasons.append("extent_below_min")
        if shape_filter_active and min_core_area_i > 0 and int(features["core16_area_px"]) < min_core_area_i:
            reasons.append("core_area_below_min")
        if min_pred_iou_f > 0.0:
            if pred_iou < 0.0:
                reasons.append("missing_pred_iou_record")
            elif pred_iou < min_pred_iou_f:
                reasons.append("pred_iou_below_min")
        if min_stability_f > 0.0:
            if stability < 0.0:
                reasons.append("missing_stability_record")
            elif stability < min_stability_f:
                reasons.append("stability_below_min")
        if not allow_relaxed and relaxed_selected:
            reasons.append("relaxed_candidate_not_durable")
        ownership_matches: list[dict[str, Any]] = []
        if anti_merge_enabled and ownership_cores:
            ownership_matches = find_multi_core_overlaps(
                mask,
                ownership_cores=ownership_cores,
                min_overlap_px=int(anti_merge_min_overlap_px_i),
                min_overlap_ratio=float(anti_merge_min_overlap_ratio_f),
            )
            matched_ids = {int(row["object_id"]) for row in ownership_matches}
            if len(matched_ids) > anti_merge_max_objects_i:
                reasons.append("multi_core_overlap")
                anti_merge_rejected_count += 1
        if reasons:
            keep[idx_i] = False
            output_only_records.append(
                {
                    "rank": int(idx_i),
                    "obj_id": int(obj_id),
                    "area_px": int(area),
                    "mask_to_uncovered_ratio": float(ratio),
                    "chosen_pred_iou": float(pred_iou),
                    "chosen_stability": float(stability),
                    "relaxed_selected": bool(relaxed_selected),
                    "bbox_area_frac": float(features["bbox_area_frac"]),
                    "edge_touch_count": int(features["edge_touch_count"]),
                    "extent": float(features["extent"]),
                    "core16_area_px": int(features["core16_area_px"]),
                    "ownership_overlap_count": int(len({int(row["object_id"]) for row in ownership_matches})),
                    "ownership_overlap_records_sample": ownership_matches[:8],
                    "reasons": reasons,
                }
            )
        admission_scores.append(
            (
                0.0 if relaxed_selected else 1.0,
                float(pred_iou if pred_iou >= 0.0 else 0.0),
                float(stability if stability >= 0.0 else 0.0),
                int(area),
            )
        )

    cap = max(0, int(max_per_frame))
    if cap > 0 and int(np.count_nonzero(keep)) > cap:
        candidates = np.flatnonzero(keep)
        ranked = sorted(
            [int(v) for v in candidates.tolist()],
            key=lambda idx: admission_scores[int(idx)],
            reverse=True,
        )
        cap_keep = set(ranked[:cap])
        for idx in candidates.tolist():
            idx_i = int(idx)
            if idx_i in cap_keep:
                continue
            keep[idx_i] = False
            record = gap_records[idx_i] if idx_i < len(gap_records) else {}
            output_only_records.append(
                {
                    "rank": int(idx_i),
                    "obj_id": int(obj_ids[idx_i]),
                    "area_px": int(areas[idx_i]),
                    "chosen_pred_iou": float(record.get("chosen_pred_iou", -1.0)),
                    "chosen_stability": float(record.get("chosen_stability", -1.0)),
                    "relaxed_selected": bool(record.get("relaxed_selected", False)),
                    "reasons": ["max_per_frame_cap"],
                }
            )

    admitted_ids = obj_ids[keep].astype(np.int64, copy=True)
    admitted_masks = masks[keep].astype(bool, copy=True)
    stats.update(
        {
            "admitted_mask_count": int(admitted_masks.shape[0]),
            "output_only_mask_count": int(input_count - admitted_masks.shape[0]),
            "admitted_obj_ids": [int(v) for v in admitted_ids.tolist()],
            "output_only_obj_ids": [int(v) for v in obj_ids[~keep].tolist()],
            "input_areas": [int(v) for v in areas.tolist()],
            "output_only_records_sample": output_only_records[:64],
            "anti_merge_rejected_count": int(anti_merge_rejected_count),
        }
    )
    return admitted_ids, admitted_masks, stats


def run_sam2_point_segment_choice(
    segmentor: Any,
    rgb: np.ndarray,
    *,
    points_yx: Any,
    region_mask: np.ndarray | None,
    points_per_batch: int,
    choice_policy: str,
    iou_threshold: float,
    stability_threshold: float,
    stability_score_offset: float,
    model_mask_thresh: float,
    box_nms_thresh: float,
    empty_ratio: float,
    apply_box_nms: bool,
    nms_score_type: str,
    relaxed_min_region_ratio: float = 0.0,
    relaxed_iou_threshold: float = 0.0,
    relaxed_stability_threshold: float = 0.0,
    relaxed_min_clipped_area: int = 0,
    small_mask_max_area: int = 0,
    small_mask_min_pred_iou: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    try:
        from sam2.utils.amg import batched_mask_to_box, calculate_stability_score
    except Exception:
        from efficient_track_anything.utils.amg import batched_mask_to_box, calculate_stability_score
    from torchvision.ops import nms

    h, w = rgb.shape[:2]
    choice_policy = str(choice_policy)
    if choice_policy not in {"largest_valid_mask_per_point", "smallest_valid_mask_per_point"}:
        raise ValueError(f"unsupported choice_policy={choice_policy}")
    if str(nms_score_type) not in {"pred_iou", "stability"}:
        raise ValueError(f"unsupported nms_score_type={nms_score_type}")

    segmentor.reset_predictor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentor.set_image(rgb)
    selected_batches = []
    selected_score_batches = []
    selected_prompt_records: list[dict[str, Any]] = []
    prompt_with_good = 0
    prompt_with_relaxed = 0
    relaxed_selected_count = 0
    raw_option_count = 0
    global_prompt_offset = 0
    batch_size = max(int(points_per_batch), 1)
    small_quality_drop_records: list[dict[str, Any]] = []
    region_ratio = 0.0
    region_t = None
    if region_mask is not None:
        region_bool = region_mask.astype(bool, copy=False)
        region_ratio = float(np.count_nonzero(region_bool)) / float(max(region_bool.size, 1))
        region_t = torch.as_tensor(region_bool, device="cuda")
    relaxed_enabled = bool(
        region_mask is not None
        and float(relaxed_min_region_ratio) > 0.0
        and region_ratio >= float(relaxed_min_region_ratio)
        and float(relaxed_iou_threshold) > 0.0
        and float(relaxed_stability_threshold) > 0.0
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, int(points_yx.shape[0]), batch_size):
            points_batch = points_yx[start : start + batch_size]
            if int(points_batch.shape[0]) == 0:
                continue
            pts_px = 0.5 * torch.tensor([h - 1, w - 1], device="cuda", dtype=torch.float32) * (points_batch + 1.0)
            pts_px = pts_px.round().long().flip(-1).float()
            coords = segmentor._transforms.transform_coords(pts_px.unsqueeze(1), normalize=True, orig_hw=(h, w))
            labels = torch.ones((points_batch.shape[0], 1), dtype=torch.int, device="cuda")
            masks, iou_predictions, _ = segmentor._predict(coords, labels, multimask_output=True, return_logits=True)
            stability = calculate_stability_score(masks, float(model_mask_thresh), float(stability_score_offset))
            good = (iou_predictions > float(iou_threshold)) & (stability >= float(stability_threshold))
            raw_option_count += int(good.numel())
            areas = (masks > float(model_mask_thresh)).sum(dim=(-1, -2), dtype=torch.int64)
            effective_good = good
            relaxed_good = torch.zeros_like(good, dtype=torch.bool)
            clipped_areas = areas
            if relaxed_enabled:
                mask_bool = masks > float(model_mask_thresh)
                if region_t is not None:
                    clipped_areas = (mask_bool & region_t.unsqueeze(0).unsqueeze(0)).sum(
                        dim=(-1, -2),
                        dtype=torch.int64,
                    )
                relaxed_good = (
                    (iou_predictions > float(relaxed_iou_threshold))
                    & (stability >= float(relaxed_stability_threshold))
                    & (clipped_areas >= int(relaxed_min_clipped_area))
                )
                strict_prompt_has_good = good.any(dim=1)
                relaxed_good = relaxed_good & ~strict_prompt_has_good.unsqueeze(1)
                effective_good = good | relaxed_good
            area_for_choice = areas.clone()
            if choice_policy == "largest_valid_mask_per_point":
                area_for_choice[~effective_good] = -1
                chosen_idx = area_for_choice.argmax(dim=1)
            else:
                area_for_choice[~effective_good] = torch.iinfo(torch.int64).max // 4
                chosen_idx = area_for_choice.argmin(dim=1)
            has_good = effective_good.any(dim=1)
            strict_has_good = good.any(dim=1)
            relaxed_has_good = relaxed_good.any(dim=1)
            prompt_with_good += int(strict_has_good.sum().item())
            prompt_with_relaxed += int(relaxed_has_good.sum().item())
            prompt_indices = torch.nonzero(has_good, as_tuple=False).flatten()
            if int(prompt_indices.numel()) > 0:
                selected = masks[prompt_indices, chosen_idx[prompt_indices]] > float(model_mask_thresh)
                if str(nms_score_type) == "pred_iou":
                    selected_scores = iou_predictions[prompt_indices, chosen_idx[prompt_indices]].float()
                else:
                    selected_scores = stability[prompt_indices, chosen_idx[prompt_indices]].float()
                selected_areas = areas[prompt_indices, chosen_idx[prompt_indices]].detach().cpu().tolist()
                selected_clipped_areas = clipped_areas[prompt_indices, chosen_idx[prompt_indices]].detach().cpu().tolist()
                selected_ious = iou_predictions[prompt_indices, chosen_idx[prompt_indices]].detach().cpu().tolist()
                selected_stabilities = stability[prompt_indices, chosen_idx[prompt_indices]].detach().cpu().tolist()
                selected_choice_indices = chosen_idx[prompt_indices].detach().cpu().tolist()
                selected_good_counts = good[prompt_indices].sum(dim=1).detach().cpu().tolist()
                selected_relaxed_counts = relaxed_good[prompt_indices].sum(dim=1).detach().cpu().tolist()
                selected_is_relaxed = relaxed_good[prompt_indices, chosen_idx[prompt_indices]].detach().cpu().tolist()
                selected_pts_px = pts_px[prompt_indices].detach().cpu().tolist()
                for local_rank, prompt_idx in enumerate(prompt_indices.detach().cpu().tolist()):
                    px = selected_pts_px[int(local_rank)]
                    relaxed_selected = bool(selected_is_relaxed[int(local_rank)])
                    if relaxed_selected:
                        relaxed_selected_count += 1
                    selected_prompt_records.append(
                        {
                            "prompt_index": int(global_prompt_offset + int(prompt_idx)),
                            "prompt_x": float(px[0]),
                            "prompt_y": float(px[1]),
                            "chosen_multimask_index": int(selected_choice_indices[int(local_rank)]),
                            "chosen_area_px": int(selected_areas[int(local_rank)]),
                            "chosen_clipped_area_px": int(selected_clipped_areas[int(local_rank)]),
                            "chosen_pred_iou": float(selected_ious[int(local_rank)]),
                            "chosen_stability": float(selected_stabilities[int(local_rank)]),
                            "good_option_count": int(selected_good_counts[int(local_rank)]),
                            "relaxed_option_count": int(selected_relaxed_counts[int(local_rank)]),
                            "relaxed_selected": relaxed_selected,
                        }
                    )
                selected_batches.append(selected)
                selected_score_batches.append(selected_scores)
            global_prompt_offset += int(points_batch.shape[0])

    if selected_batches:
        selected_t = torch.cat(selected_batches, dim=0)
        selected_scores_t = torch.cat(selected_score_batches, dim=0)
        quality_max_area = max(0, int(small_mask_max_area))
        quality_min_iou = float(small_mask_min_pred_iou)
        if quality_max_area > 0 and quality_min_iou > 0.0 and selected_prompt_records:
            keep_indices: list[int] = []
            for record_idx, record in enumerate(selected_prompt_records):
                area_key = "chosen_clipped_area_px" if region_t is not None else "chosen_area_px"
                candidate_area = int(record.get(area_key, record.get("chosen_area_px", 0)) or 0)
                pred_iou = float(record.get("chosen_pred_iou", 0.0) or 0.0)
                drop = bool(candidate_area <= quality_max_area and pred_iou < quality_min_iou)
                if drop:
                    small_quality_drop_records.append(
                        {
                            "prompt_index": int(record.get("prompt_index", -1)),
                            "chosen_area_px": int(record.get("chosen_area_px", 0) or 0),
                            "chosen_clipped_area_px": int(record.get("chosen_clipped_area_px", 0) or 0),
                            "chosen_pred_iou": float(pred_iou),
                            "chosen_stability": float(record.get("chosen_stability", 0.0) or 0.0),
                            "relaxed_selected": bool(record.get("relaxed_selected", False)),
                            "small_mask_max_area": int(quality_max_area),
                            "small_mask_min_pred_iou": float(quality_min_iou),
                        }
                    )
                else:
                    keep_indices.append(int(record_idx))
            if len(keep_indices) != len(selected_prompt_records):
                if keep_indices:
                    keep_t = torch.as_tensor(keep_indices, device=selected_t.device, dtype=torch.long)
                    selected_t = selected_t[keep_t]
                    selected_scores_t = selected_scores_t[keep_t]
                    selected_prompt_records = [selected_prompt_records[idx] for idx in keep_indices]
                else:
                    selected_t = selected_t[:0]
                    selected_scores_t = selected_scores_t[:0]
                    selected_prompt_records = []
        if region_t is not None:
            selected_t = selected_t & region_t.unsqueeze(0)
        pre_nms_count = int(selected_t.shape[0])
        if apply_box_nms and int(selected_t.shape[0]) > 0:
            boxes = batched_mask_to_box(selected_t).float()
            keep = nms(boxes, selected_scores_t, iou_threshold=float(box_nms_thresh))
            selected_t = selected_t[keep]
            selected_prompt_records = [selected_prompt_records[int(i)] for i in keep.detach().cpu().tolist()]
        pre_disjoint_selected_prompt_records = [dict(row) for row in selected_prompt_records]
        selected_np = selected_t.detach().cpu().numpy().astype(bool)
    else:
        pre_nms_count = 0
        selected_np = np.zeros((0, h, w), dtype=bool)
        pre_disjoint_selected_prompt_records = []

    disjoint_np, post_disjoint_selected_prompt_records = disjoin_smallest_first_with_records(
        selected_np,
        pre_disjoint_selected_prompt_records,
        h,
        w,
        empty_ratio=float(empty_ratio),
        fix_small_regions=True,
    )
    stats = {
        "choice_policy": choice_policy,
        "iou_threshold": float(iou_threshold),
        "stability_threshold": float(stability_threshold),
        "raw_multimask_option_count": int(raw_option_count),
        "prompt_with_good_mask_count": int(prompt_with_good),
        "prompt_with_relaxed_mask_count": int(prompt_with_relaxed),
        "relaxed_selected_mask_count": int(relaxed_selected_count),
        "relaxed_enabled": bool(relaxed_enabled),
        "relaxed_min_region_ratio": float(relaxed_min_region_ratio),
        "relaxed_iou_threshold": float(relaxed_iou_threshold),
        "relaxed_stability_threshold": float(relaxed_stability_threshold),
        "relaxed_min_clipped_area": int(relaxed_min_clipped_area),
        "region_ratio": float(region_ratio),
        "small_mask_quality_filter": {
            "enabled": bool(int(small_mask_max_area) > 0 and float(small_mask_min_pred_iou) > 0.0),
            "small_mask_max_area": int(max(0, int(small_mask_max_area))),
            "small_mask_min_pred_iou": float(small_mask_min_pred_iou),
            "dropped_mask_count": int(len(small_quality_drop_records)),
            "dropped_mask_records_sample": small_quality_drop_records[:64],
        },
        "pre_nms_mask_count": int(pre_nms_count),
        "post_disjoint_mask_count": int(disjoint_np.shape[0]),
        "apply_box_nms": bool(apply_box_nms),
        "nms_score_type": str(nms_score_type),
        "pre_disjoint_selected_prompt_records": pre_disjoint_selected_prompt_records[:64],
        "post_disjoint_selected_prompt_records": post_disjoint_selected_prompt_records,
    }
    if apply_box_nms:
        stats["post_nms_mask_count"] = int(selected_np.shape[0])
    return disjoint_np, stats


def uncovered_from_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    if masks.size:
        return ~np.any(masks.astype(bool), axis=0)
    return np.ones((h, w), dtype=bool)


def assign_gap_ids_with_recent_reuse(
    gap_masks: np.ndarray,
    *,
    current_ids: np.ndarray,
    next_obj_id: int,
    recent_mask_by_obj: dict[int, np.ndarray],
    recent_frame_by_obj: dict[int, int],
    current_frame_idx: int,
    reuse_window_frames: int,
    reuse_iou_threshold: float,
    reuse_min_area: int,
    large_reuse_iou_threshold: float = 0.0,
    large_reuse_min_area: int = 0,
    large_reuse_max_area_ratio: float = 0.0,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    """Assign gap ids, optionally reusing a recently lost object id by mask IoU."""
    if gap_masks.size == 0:
        return np.zeros((0,), dtype=np.int64), int(next_obj_id), []
    window = max(0, int(reuse_window_frames))
    iou_threshold = float(reuse_iou_threshold)
    min_area = max(0, int(reuse_min_area))
    large_iou_threshold = max(0.0, float(large_reuse_iou_threshold))
    large_min_area = max(0, int(large_reuse_min_area))
    large_max_area_ratio = max(0.0, float(large_reuse_max_area_ratio))
    if window <= 0 or (iou_threshold <= 0.0 and large_iou_threshold <= 0.0):
        return np.arange(int(next_obj_id), int(next_obj_id) + int(gap_masks.shape[0]), dtype=np.int64), int(next_obj_id) + int(gap_masks.shape[0]), []
    candidate_min_area_values = []
    if iou_threshold > 0.0:
        candidate_min_area_values.append(min_area)
    if large_iou_threshold > 0.0:
        candidate_min_area_values.append(large_min_area)
    candidate_min_area = min(candidate_min_area_values) if candidate_min_area_values else 0
    current_id_set = {int(v) for v in current_ids.tolist()} if current_ids.size else set()
    candidate_ids: list[int] = []
    for obj_id, last_frame in recent_frame_by_obj.items():
        obj_id_i = int(obj_id)
        age = int(current_frame_idx) - int(last_frame)
        if age <= 0 or age > window or obj_id_i in current_id_set:
            continue
        prev_mask = recent_mask_by_obj.get(obj_id_i)
        if prev_mask is None or int(np.count_nonzero(prev_mask)) < candidate_min_area:
            continue
        candidate_ids.append(obj_id_i)

    assigned: list[int] = []
    events: list[dict[str, Any]] = []
    used_candidates: set[int] = set()
    for gap_rank, gap_mask_raw in enumerate(gap_masks.astype(bool, copy=False)):
        gap_area = int(np.count_nonzero(gap_mask_raw))
        best_obj_id: int | None = None
        best_iou = 0.0
        best_intersection = 0
        best_union = 0
        best_prev_area = 0
        if gap_area >= candidate_min_area:
            for obj_id in candidate_ids:
                if int(obj_id) in used_candidates:
                    continue
                prev_mask = recent_mask_by_obj.get(int(obj_id))
                if prev_mask is None:
                    continue
                prev_bool = prev_mask.astype(bool, copy=False)
                prev_area = int(np.count_nonzero(prev_bool))
                inter = int(np.count_nonzero(gap_mask_raw & prev_bool))
                if inter <= 0:
                    continue
                union = int(gap_area + prev_area - inter)
                if union <= 0:
                    continue
                iou = float(inter) / float(union)
                if iou > best_iou:
                    best_iou = iou
                    best_obj_id = int(obj_id)
                    best_intersection = inter
                    best_union = union
                    best_prev_area = int(prev_area)
        effective_iou_threshold = float(iou_threshold if iou_threshold > 0.0 else 1.0e9)
        reuse_mode = "recent_reuse"
        area_ratio = 0.0
        large_pair_eligible = False
        if best_obj_id is not None and gap_area > 0 and best_prev_area > 0:
            area_ratio = max(float(gap_area) / float(best_prev_area), float(best_prev_area) / float(gap_area))
            large_pair_eligible = bool(
                large_iou_threshold > 0.0
                and large_min_area > 0
                and gap_area >= large_min_area
                and best_prev_area >= large_min_area
                and (large_max_area_ratio <= 0.0 or area_ratio <= large_max_area_ratio)
            )
            if large_pair_eligible and large_iou_threshold < effective_iou_threshold:
                effective_iou_threshold = float(large_iou_threshold)
                reuse_mode = "large_temporal_reuse"
        if best_obj_id is not None and best_iou >= effective_iou_threshold:
            assigned_id = int(best_obj_id)
            used_candidates.add(assigned_id)
            events.append(
                {
                    "gap_rank": int(gap_rank),
                    "assigned_object_id": int(assigned_id),
                    "source": str(reuse_mode),
                    "reuse_iou": float(best_iou),
                    "effective_iou_threshold": float(effective_iou_threshold),
                    "base_reuse_iou_threshold": float(iou_threshold),
                    "large_reuse_iou_threshold": float(large_iou_threshold),
                    "large_pair_eligible": bool(large_pair_eligible),
                    "area_ratio": float(area_ratio),
                    "intersection_px": int(best_intersection),
                    "union_px": int(best_union),
                    "gap_area_px": int(gap_area),
                    "previous_area_px": int(best_prev_area),
                    "previous_frame_idx": int(recent_frame_by_obj.get(assigned_id, -1)),
                    "age_frames": int(current_frame_idx) - int(recent_frame_by_obj.get(assigned_id, current_frame_idx)),
                    "reuse_window_frames": int(window),
                    "reuse_iou_threshold": float(effective_iou_threshold),
                }
            )
        else:
            assigned_id = int(next_obj_id)
            next_obj_id += 1
        assigned.append(int(assigned_id))
    return np.asarray(assigned, dtype=np.int64), int(next_obj_id), events


def canonicalize_output_ids_with_recent_overlap(
    obj_ids: np.ndarray,
    masks: np.ndarray,
    *,
    recent_mask_by_obj: dict[int, np.ndarray],
    recent_frame_by_obj: dict[int, int],
    current_frame_idx: int,
    reuse_window_frames: int,
    reuse_iou_threshold: float,
    reuse_min_area: int,
    canonical_preference: str = "lower_id",
    prevent_collision_union: bool = False,
    large_reuse_iou_threshold: float = 0.0,
    large_reuse_min_area: int = 0,
    large_reuse_max_area_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Stabilize output labels by folding high-IoU recent aliases to a canonical id."""
    if masks.size == 0 or obj_ids.size == 0:
        return obj_ids, masks, []
    window = max(0, int(reuse_window_frames))
    iou_threshold = float(reuse_iou_threshold)
    min_area = max(0, int(reuse_min_area))
    large_iou_threshold = max(0.0, float(large_reuse_iou_threshold))
    large_min_area = max(0, int(large_reuse_min_area))
    large_max_area_ratio = max(0.0, float(large_reuse_max_area_ratio))
    if window <= 0 or (iou_threshold <= 0.0 and large_iou_threshold <= 0.0):
        return obj_ids, masks, []
    preference = str(canonical_preference or "lower_id").strip().lower()
    candidate_min_area_values = []
    if iou_threshold > 0.0:
        candidate_min_area_values.append(min_area)
    if large_iou_threshold > 0.0:
        candidate_min_area_values.append(large_min_area)
    candidate_min_area = min(candidate_min_area_values) if candidate_min_area_values else 0

    candidate_ids: list[int] = []
    for obj_id, last_frame in recent_frame_by_obj.items():
        age = int(current_frame_idx) - int(last_frame)
        if age <= 0 or age > window:
            continue
        prev_mask = recent_mask_by_obj.get(int(obj_id))
        if prev_mask is None or int(np.count_nonzero(prev_mask)) < candidate_min_area:
            continue
        candidate_ids.append(int(obj_id))

    relabel_events: list[dict[str, Any]] = []
    candidate_relabel_events: dict[int, dict[str, Any]] = {}
    canonical_ids: list[int] = []
    original_ids: list[int] = []
    mask_areas: list[int] = []
    match_ious: list[float] = []
    match_thresholds: list[float] = []
    match_modes: list[str] = []
    match_area_ratios: list[float] = []
    for rank, (obj_id_raw, mask_raw) in enumerate(zip(obj_ids.tolist(), masks.astype(bool), strict=False)):
        obj_id = int(obj_id_raw)
        original_ids.append(obj_id)
        mask = mask_raw.astype(bool, copy=False)
        area = int(np.count_nonzero(mask))
        mask_areas.append(area)
        best_obj_id: int | None = None
        best_iou = 0.0
        best_intersection = 0
        best_union = 0
        best_prev_area = 0
        if area >= candidate_min_area:
            for candidate_id in candidate_ids:
                if int(candidate_id) == obj_id:
                    continue
                prev_mask = recent_mask_by_obj.get(int(candidate_id))
                if prev_mask is None:
                    continue
                prev_bool = prev_mask.astype(bool, copy=False)
                prev_area = int(np.count_nonzero(prev_bool))
                inter = int(np.count_nonzero(mask & prev_bool))
                if inter <= 0:
                    continue
                union = int(area + prev_area - inter)
                if union <= 0:
                    continue
                iou = float(inter) / float(union)
                if iou > best_iou:
                    best_iou = iou
                    best_obj_id = int(candidate_id)
                    best_intersection = inter
                    best_union = union
                    best_prev_area = int(prev_area)
        effective_iou_threshold = float(iou_threshold if iou_threshold > 0.0 else 1.0e9)
        reuse_mode = "recent_reuse"
        area_ratio = 0.0
        large_pair_eligible = False
        if best_obj_id is not None and area > 0 and best_prev_area > 0:
            area_ratio = max(float(area) / float(best_prev_area), float(best_prev_area) / float(area))
            large_pair_eligible = bool(
                large_iou_threshold > 0.0
                and large_min_area > 0
                and area >= large_min_area
                and best_prev_area >= large_min_area
                and (large_max_area_ratio <= 0.0 or area_ratio <= large_max_area_ratio)
            )
            if large_pair_eligible and large_iou_threshold < effective_iou_threshold:
                effective_iou_threshold = float(large_iou_threshold)
                reuse_mode = "large_temporal_reuse"
        canonical_id = obj_id
        if best_obj_id is not None and best_iou >= effective_iou_threshold:
            if preference in {"recent_id", "previous_id", "temporal"}:
                canonical_id = int(best_obj_id)
            else:
                canonical_id = min(obj_id, int(best_obj_id))
        canonical_ids.append(int(canonical_id))
        match_ious.append(float(best_iou))
        match_thresholds.append(float(effective_iou_threshold))
        match_modes.append(str(reuse_mode))
        match_area_ratios.append(float(area_ratio))
        if canonical_id != obj_id:
            candidate_relabel_events[int(rank)] = {
                "mask_rank": int(rank),
                "original_object_id": int(obj_id),
                "canonical_object_id": int(canonical_id),
                "matched_recent_object_id": int(best_obj_id),
                "reuse_iou": float(best_iou),
                "effective_iou_threshold": float(effective_iou_threshold),
                "base_reuse_iou_threshold": float(iou_threshold),
                "large_reuse_iou_threshold": float(large_iou_threshold),
                "large_pair_eligible": bool(large_pair_eligible),
                "reuse_mode": str(reuse_mode),
                "area_ratio": float(area_ratio),
                "intersection_px": int(best_intersection),
                "union_px": int(best_union),
                "mask_area_px": int(area),
                "previous_area_px": int(best_prev_area),
                "previous_frame_idx": int(recent_frame_by_obj.get(int(best_obj_id), -1)),
                "age_frames": int(current_frame_idx)
                - int(recent_frame_by_obj.get(int(best_obj_id), current_frame_idx)),
                "reuse_window_frames": int(window),
                "reuse_iou_threshold": float(effective_iou_threshold),
                "canonical_preference": str(preference),
            }

    collision_guard_events: list[dict[str, Any]] = []
    if bool(prevent_collision_union) and canonical_ids:
        ranks_by_canonical: dict[int, list[int]] = {}
        for rank, canonical_id in enumerate(canonical_ids):
            ranks_by_canonical.setdefault(int(canonical_id), []).append(int(rank))
        for canonical_id, ranks in ranks_by_canonical.items():
            if len(ranks) <= 1:
                continue

            def _collision_score(rank: int) -> tuple[int, float, int, int]:
                same_id_bonus = 1 if int(original_ids[rank]) == int(canonical_id) else 0
                return (
                    same_id_bonus,
                    float(match_ious[rank]),
                    int(mask_areas[rank]),
                    -int(rank),
                )

            keep_rank = max(ranks, key=_collision_score)
            for rank in ranks:
                if int(rank) == int(keep_rank):
                    continue
                if int(canonical_ids[rank]) == int(original_ids[rank]):
                    continue
                collision_guard_events.append(
                    {
                        "action": "collision_guard_revert",
                        "canonical_object_id": int(canonical_id),
                        "kept_mask_rank": int(keep_rank),
                        "kept_original_object_id": int(original_ids[keep_rank]),
                        "kept_reuse_iou": float(match_ious[keep_rank]),
                        "kept_effective_iou_threshold": float(match_thresholds[keep_rank]),
                        "kept_reuse_mode": str(match_modes[keep_rank]),
                        "kept_area_ratio": float(match_area_ratios[keep_rank]),
                        "kept_mask_area_px": int(mask_areas[keep_rank]),
                        "blocked_mask_rank": int(rank),
                        "blocked_original_object_id": int(original_ids[rank]),
                        "blocked_reuse_iou": float(match_ious[rank]),
                        "blocked_effective_iou_threshold": float(match_thresholds[rank]),
                        "blocked_reuse_mode": str(match_modes[rank]),
                        "blocked_area_ratio": float(match_area_ratios[rank]),
                        "blocked_mask_area_px": int(mask_areas[rank]),
                        "policy": "prevent_collision_union_keep_same_id_then_best_iou",
                    }
                )
                canonical_ids[rank] = int(original_ids[rank])

    for rank, event in candidate_relabel_events.items():
        if int(canonical_ids[int(rank)]) != int(original_ids[int(rank)]):
            relabel_events.append(event)
    relabel_events.extend(collision_guard_events)

    merged: dict[int, np.ndarray] = {}
    ordered_ids: list[int] = []
    for canonical_id, mask in zip(canonical_ids, masks.astype(bool), strict=False):
        canonical_i = int(canonical_id)
        if canonical_i not in merged:
            merged[canonical_i] = mask.astype(bool, copy=True)
            ordered_ids.append(canonical_i)
        else:
            merged[canonical_i] |= mask.astype(bool, copy=False)
    if not ordered_ids:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,) + masks.shape[-2:], dtype=bool), relabel_events
    out_masks = np.stack([merged[int(obj_id)] for obj_id in ordered_ids], axis=0).astype(bool)
    out_ids = np.asarray(ordered_ids, dtype=np.int64)
    return out_ids, out_masks, relabel_events


def merge_or_suppress_output_fragments(
    obj_ids: np.ndarray,
    masks: np.ndarray,
    h: int,
    w: int,
    *,
    fragment_max_area: int,
    suppress_max_area: int,
    merge_dilate_px: int,
    merge_min_touch_px: int,
    merge_min_touch_ratio: float,
    merge_min_neighbor_area: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Clean tiny disconnected output components without mutating SAM2 state."""
    if masks.size == 0 or obj_ids.size == 0:
        return obj_ids, masks, []
    max_area = max(0, int(fragment_max_area))
    suppress_area = max(0, int(suppress_max_area))
    if max_area <= 0 and suppress_area <= 0:
        return obj_ids, masks, []
    radius = max(1, int(merge_dilate_px))
    min_touch = max(0, int(merge_min_touch_px))
    min_touch_ratio = max(0.0, float(merge_min_touch_ratio))
    min_neighbor_area = max(0, int(merge_min_neighbor_area))

    label = label_from_id_masks(obj_ids, masks, h, w).astype(np.int32, copy=False)
    if not np.any(label):
        return np.zeros((0,), dtype=np.int64), np.zeros((0, h, w), dtype=bool), []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    events: list[dict[str, Any]] = []

    for label_value in [int(v) for v in np.unique(label) if int(v) > 0]:
        current = label == int(label_value)
        if not np.any(current):
            continue
        n_components, components, stats, _ = cv2.connectedComponentsWithStats(current.astype(np.uint8), 8)
        if int(n_components) <= 2:
            continue
        for component_id in range(1, int(n_components)):
            component_area = int(stats[component_id, cv2.CC_STAT_AREA])
            can_merge = bool(max_area > 0 and component_area <= max_area)
            can_suppress = bool(suppress_area > 0 and component_area <= suppress_area)
            if not can_merge and not can_suppress:
                continue
            component = components == int(component_id)
            action = ""
            target_label = 0
            touch_px = 0
            touch_ratio = 0.0
            if can_merge:
                border = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool) & ~component
                neighbor_values, neighbor_counts = np.unique(label[border], return_counts=True)
                best_count = 0
                best_label = 0
                for raw_value, raw_count in zip(neighbor_values.tolist(), neighbor_counts.tolist(), strict=False):
                    candidate_label = int(raw_value)
                    if candidate_label <= 0 or candidate_label == int(label_value):
                        continue
                    candidate_area = int(np.count_nonzero(label == candidate_label))
                    if candidate_area < min_neighbor_area:
                        continue
                    if int(raw_count) > best_count:
                        best_count = int(raw_count)
                        best_label = int(candidate_label)
                touch_px = int(best_count)
                touch_ratio = float(touch_px) / float(max(component_area, 1))
                if best_label > 0 and touch_px >= min_touch and touch_ratio >= min_touch_ratio:
                    label[component] = int(best_label)
                    action = "merge_to_neighbor"
                    target_label = int(best_label)
            if not action and can_suppress:
                label[component] = 0
                action = "suppress_to_background"
            if action:
                events.append(
                    {
                        "source_object_id": int(label_value) - 1,
                        "component_id": int(component_id),
                        "component_area_px": int(component_area),
                        "action": action,
                        "target_object_id": int(target_label) - 1 if target_label > 0 else -1,
                        "touch_px": int(touch_px),
                        "touch_ratio": float(touch_ratio),
                        "fragment_max_area": int(max_area),
                        "suppress_max_area": int(suppress_area),
                    }
                )

    present_labels = [int(v) for v in np.unique(label) if int(v) > 0]
    if not present_labels:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, h, w), dtype=bool), events
    original_order = [int(v) + 1 for v in obj_ids.tolist()]
    ordered_labels = [value for value in original_order if value in set(present_labels)]
    ordered_labels.extend(value for value in sorted(present_labels) if value not in set(ordered_labels))
    out_masks = np.stack([label == int(value) for value in ordered_labels], axis=0).astype(bool)
    out_ids = np.asarray([int(value) - 1 for value in ordered_labels], dtype=np.int64)
    return out_ids, out_masks, events


def sample_component_adaptive_points_yx(
    mask_np: np.ndarray,
    *,
    max_points: int,
    min_component_area: int,
    base_points_per_component: int,
    area_per_extra_point: int,
    max_points_per_component: int,
    min_image_edge_distance_px: int = 0,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch

    mask = mask_np.astype(bool)
    h, w = mask.shape
    max_points = max(int(max_points), 0)
    edge_margin = max(0, int(min_image_edge_distance_px))
    if max_points <= 0 or not np.any(mask):
        return torch.zeros((0, 2), device="cuda", dtype=torch.float32), {
            "sampler": "component_adaptive",
            "component_count": 0,
            "kept_component_count": 0,
            "point_count": 0,
            "max_points": int(max_points),
            "min_image_edge_distance_px": int(edge_margin),
        }

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components: list[tuple[int, int]] = []
    for label_id in range(1, int(n_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= int(min_component_area):
            components.append((label_id, area))
    components.sort(key=lambda item: item[1], reverse=True)

    rng = np.random.default_rng(int(seed))
    points: list[tuple[float, float]] = []
    per_component_counts: list[dict[str, int]] = []
    point_records: list[dict[str, Any]] = []
    area_per_extra_point = max(int(area_per_extra_point), 1)
    yy = np.arange(h, dtype=np.int32)[:, None]
    xx = np.arange(w, dtype=np.int32)[None, :]
    image_edge_distance = np.minimum(
        np.minimum(yy, h - 1 - yy),
        np.minimum(xx, w - 1 - xx),
    ).astype(np.float32)
    for label_id, area in components:
        if len(points) >= max_points:
            break
        n_for_component = int(base_points_per_component) + int(area // area_per_extra_point)
        n_for_component = max(1, min(int(max_points_per_component), n_for_component))
        n_for_component = min(n_for_component, max_points - len(points))
        comp = labels == int(label_id)
        dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.nonzero(comp)
        x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        edge_eligible = comp & (image_edge_distance >= float(edge_margin))
        edge_margin_fallback = bool(edge_margin > 0 and not np.any(edge_eligible))
        sampling_region = comp if edge_margin_fallback else edge_eligible
        sample_ys, sample_xs = np.nonzero(sampling_region)
        chosen: list[tuple[int, int]] = []
        if edge_margin > 0:
            score = np.minimum(dist, image_edge_distance)
            if not edge_margin_fallback:
                score[~edge_eligible] = 0.0
        else:
            score = dist.copy()
        initial_max_score = float(np.max(score)) if score.size else 0.0
        for _ in range(n_for_component):
            if not np.isfinite(score).any() or float(score.max()) <= 0.0:
                pool_ys, pool_xs = (sample_ys, sample_xs) if sample_ys.size else (ys, xs)
                idx = int(rng.integers(0, len(pool_ys)))
                y, x = int(pool_ys[idx]), int(pool_xs[idx])
            else:
                y, x = [int(v) for v in np.unravel_index(int(np.argmax(score)), score.shape)]
            chosen.append((y, x))
            radius = max(6, int(round(np.sqrt(float(area) / float(max(n_for_component, 1))) * 0.35)))
            cv2.circle(score, (x, y), radius, 0.0, thickness=-1)
        for y, x in chosen:
            y_norm = 2.0 * float(y) / float(max(h - 1, 1)) - 1.0
            x_norm = 2.0 * float(x) / float(max(w - 1, 1)) - 1.0
            points.append((y_norm, x_norm))
            point_records.append(
                {
                    "point_index": int(len(points) - 1),
                    "component_label": int(label_id),
                    "component_area": int(area),
                    "component_bbox_xywh": [int(x0), int(y0), int(bw), int(bh)],
                    "x": int(x),
                    "y": int(y),
                    "x_norm": float(x_norm),
                    "y_norm": float(y_norm),
                    "distance_to_component_boundary_px": float(dist[int(y), int(x)]),
                    "distance_to_image_edge_px": float(min(int(x), int(y), int(w - 1 - x), int(h - 1 - y))),
                    "safe_interior_score_px": float(
                        min(
                            float(dist[int(y), int(x)]),
                            float(min(int(x), int(y), int(w - 1 - x), int(h - 1 - y))),
                        )
                    )
                    if edge_margin > 0
                    else float(dist[int(y), int(x)]),
                    "min_image_edge_distance_px": int(edge_margin),
                    "edge_margin_fallback": bool(edge_margin_fallback),
                }
            )
        per_component_counts.append(
            {
                "label": int(label_id),
                "area": int(area),
                "bbox_xywh": [int(x0), int(y0), int(bw), int(bh)],
                "edge_eligible_area": int(np.count_nonzero(edge_eligible)),
                "edge_margin_fallback": bool(edge_margin_fallback),
                "points": int(len(chosen)),
                "max_distance_to_boundary_px": float(np.max(dist)) if dist.size else 0.0,
                "max_safe_interior_score_px": float(initial_max_score),
            }
        )

    if points:
        pts = torch.tensor(points, device="cuda", dtype=torch.float32)
    else:
        pts = torch.zeros((0, 2), device="cuda", dtype=torch.float32)
    meta = {
        "sampler": "component_adaptive",
        "component_count": int(max(n_labels - 1, 0)),
        "kept_component_count": int(len(components)),
        "point_count": int(pts.shape[0]),
        "max_points": int(max_points),
        "min_component_area": int(min_component_area),
        "base_points_per_component": int(base_points_per_component),
        "area_per_extra_point": int(area_per_extra_point),
        "max_points_per_component": int(max_points_per_component),
        "min_image_edge_distance_px": int(edge_margin),
        "per_component_counts": per_component_counts[:32],
        "point_records": point_records[:128],
    }
    return pts, meta


def propagate_new_masks_chunked(
    predictor: Any,
    *,
    tracker: str,
    video_dir: Path,
    seed_frame: int,
    obj_ids: np.ndarray,
    masks: np.ndarray,
    total_frames: int,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    chunk_size: int,
    feature_bank_frame_offset: int = 0,
    state_num_frames_override: int | None = None,
    clear_cached_features_after_init: bool = False,
    video_state_template: dict[str, Any] | None = None,
    chunk_runtime_records: list[dict[str, Any]] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    import gc
    import torch

    if masks.size == 0:
        return {}
    n_masks = int(masks.shape[0])
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or n_masks <= chunk_size:
        chunk_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = propagate_new_masks(
                predictor,
                tracker=tracker,
                video_dir=video_dir,
                seed_frame=seed_frame,
                obj_ids=obj_ids,
                masks=masks,
                total_frames=total_frames,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                feature_bank_frame_offset=int(feature_bank_frame_offset),
                state_num_frames_override=state_num_frames_override,
                clear_cached_features_after_init=bool(clear_cached_features_after_init),
                video_state_template=video_state_template,
            )
        if chunk_runtime_records is not None:
            chunk_runtime_records.append(
                {
                    "chunk_start": 0,
                    "chunk_end": int(n_masks),
                    "chunk_object_count": int(n_masks),
                    "chunk_runtime_sec": float(time.time() - chunk_t0),
                    "chunk_future_output_mask_count": int(sum(len(v) for v in out.values())),
                }
            )
        return out

    merged: dict[int, dict[int, np.ndarray]] = {}
    for start in range(0, n_masks, chunk_size):
        end = min(start + chunk_size, n_masks)
        chunk_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            partial = propagate_new_masks(
                predictor,
                tracker=tracker,
                video_dir=video_dir,
                seed_frame=seed_frame,
                obj_ids=obj_ids[start:end],
                masks=masks[start:end],
                total_frames=total_frames,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu,
                feature_bank_frame_offset=int(feature_bank_frame_offset),
                state_num_frames_override=state_num_frames_override,
                clear_cached_features_after_init=bool(clear_cached_features_after_init),
                video_state_template=video_state_template,
            )
        if chunk_runtime_records is not None:
            chunk_runtime_records.append(
                {
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                    "chunk_object_count": int(end - start),
                    "chunk_runtime_sec": float(time.time() - chunk_t0),
                    "chunk_future_output_mask_count": int(sum(len(v) for v in partial.values())),
                }
            )
        for frame_idx, frame_outputs in partial.items():
            merged.setdefault(int(frame_idx), {}).update(frame_outputs)
        gc.collect()
        torch.cuda.empty_cache()
    return merged


def add_masks_to_stream_state(
    predictor: Any,
    state: dict[str, Any],
    *,
    tracker: str,
    frame_idx: int,
    obj_ids: np.ndarray,
    masks: np.ndarray,
) -> None:
    """Add masks to a persistent SAM2 state, allowing new ids after tracking starts."""
    import torch
    from contextlib import nullcontext

    if masks.size == 0:
        return
    autocast_dtype = torch.float32
    try:
        autocast_dtype = next(predictor.parameters()).dtype
    except Exception:
        pass
    old_obj_count = len(state.get("obj_ids", []))
    adding_new_ids_after_tracking = bool(state.get("tracking_has_started", False)) and any(
        int(obj_id) not in state.get("obj_id_to_idx", {}) for obj_id in obj_ids.tolist()
    )
    if adding_new_ids_after_tracking:
        # Treat a post-start new object as a fresh conditioning input on this frame.
        state.get("frames_already_tracked", {}).pop(int(frame_idx), None)
    old_tracking_started = bool(state.get("tracking_has_started", False))
    state["tracking_has_started"] = False
    try:
        for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
            if tracker == "sam2":
                mask_arg: Any = torch.from_numpy(mask.astype(np.float32))
            else:
                mask_arg = mask.astype(np.float32)
            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if tracker == "sam2" and autocast_dtype in {torch.bfloat16, torch.float16}
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                    mask=mask_arg,
                )
    finally:
        state["tracking_has_started"] = old_tracking_started
    new_obj_count = len(state.get("obj_ids", []))
    if old_tracking_started and new_obj_count > old_obj_count:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            reconsolidate_stream_state_outputs(predictor, state)


def reconsolidate_stream_state_outputs(predictor: Any, state: dict[str, Any]) -> None:
    """Rebuild packed SAM2 outputs after post-start object births.

    SAM2 stores packed tensors with shape [num_objects, ...]. If a new object is
    added after tracking has started, older packed outputs still have the old
    object count; subsequent memory attention then fails when stacking obj_ptrs.
    Re-consolidating through SAM2's per-object stores pads missing objects with
    the model's own empty-object placeholders and keeps streaming inference usable.
    """
    output_dict = state.get("output_dict", {})
    for storage_key, is_cond in (
        ("cond_frame_outputs", True),
        ("non_cond_frame_outputs", False),
    ):
        frame_indices = sorted(output_dict.get(storage_key, {}).keys())
        for prior_frame_idx in frame_indices:
            consolidated_out = predictor._consolidate_temp_output_across_obj(
                state,
                int(prior_frame_idx),
                is_cond=bool(is_cond),
                run_mem_encoder=True,
            )
            output_dict[storage_key][int(prior_frame_idx)] = consolidated_out
            predictor._add_output_per_object(
                state,
                int(prior_frame_idx),
                consolidated_out,
                storage_key,
            )


def infer_stream_frame(
    predictor: Any,
    state: dict[str, Any],
    *,
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if hasattr(predictor, "infer_single_frame"):
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.infer_single_frame(state, int(frame_idx))
        else:
            iterator = predictor.propagate_in_video(
                state,
                start_frame_idx=int(frame_idx),
                max_frame_num_to_track=0,
            )
            out_frame_idx, out_obj_ids, out_mask_logits = next(iterator)
    if int(out_frame_idx) != int(frame_idx):
        raise RuntimeError(f"streaming inference returned frame {out_frame_idx}, expected {frame_idx}")
    masks = (out_mask_logits > 0.0).detach().cpu().numpy().squeeze(1).astype(bool)
    ids = np.asarray([int(v) for v in out_obj_ids], dtype=np.int64)
    if STREAM_INFER_TRACE_HOOK is not None:
        STREAM_INFER_TRACE_HOOK(int(frame_idx), ids, out_mask_logits, masks)
    return ids, masks


def prune_stream_noncond_memory(
    state: dict[str, Any],
    *,
    current_frame_idx: int,
    keep_recent_frames: int,
) -> list[int]:
    """Drop old non-conditioning outputs that SAM2 no longer needs for forward streaming."""
    keep_recent = int(keep_recent_frames)
    if keep_recent <= 0:
        return []
    min_keep_frame = int(current_frame_idx) - keep_recent + 1
    output_dict = state.get("output_dict", {})
    noncond_outputs = output_dict.get("non_cond_frame_outputs", {})
    remove_frames = sorted(int(idx) for idx in list(noncond_outputs.keys()) if int(idx) < min_keep_frame)
    if not remove_frames:
        return []
    consolidated = state.get("consolidated_frame_inds", {}).get("non_cond_frame_outputs", set())
    per_obj = state.get("output_dict_per_obj", {})
    temp_per_obj = state.get("temp_output_dict_per_obj", {})
    frames_already_tracked = state.get("frames_already_tracked", {})
    for frame_idx in remove_frames:
        noncond_outputs.pop(frame_idx, None)
        consolidated.discard(frame_idx)
        frames_already_tracked.pop(frame_idx, None)
        for obj_output in per_obj.values():
            obj_output.get("non_cond_frame_outputs", {}).pop(frame_idx, None)
        for obj_output in temp_per_obj.values():
            obj_output.get("non_cond_frame_outputs", {}).pop(frame_idx, None)
    return remove_frames


def prune_stream_invisible_objects(
    predictor: Any,
    state: dict[str, Any],
    *,
    current_frame_idx: int,
    visible_ids: np.ndarray,
    visible_masks: np.ndarray,
    last_visible_frame_by_obj: dict[int, int],
    max_visible_area_by_obj: dict[int, int],
    min_visible_area: int,
    invisible_after_frames: int,
    protect_min_ever_area: int = 0,
    protect_max_objects: int = 0,
) -> tuple[list[int], list[int]]:
    """Remove inactive object ids through SAM2's public state mutation helper."""
    after = int(invisible_after_frames)
    if after <= 0 or not hasattr(predictor, "remove_object"):
        return [], []
    min_area = max(0, int(min_visible_area))
    protect_area = max(0, int(protect_min_ever_area))
    visible_now: set[int] = set()
    if visible_ids.size and visible_masks.size:
        for obj_id, mask in zip(visible_ids.tolist(), visible_masks.astype(bool), strict=False):
            area_i = int(np.count_nonzero(mask))
            obj_id_i = int(obj_id)
            max_visible_area_by_obj[obj_id_i] = max(int(max_visible_area_by_obj.get(obj_id_i, 0)), area_i)
            if area_i >= min_area:
                visible_now.add(obj_id_i)
                last_visible_frame_by_obj[obj_id_i] = int(current_frame_idx)

    pruned: list[int] = []
    expired_unprotected: list[int] = []
    expired_protected_candidates: list[tuple[int, int]] = []
    for obj_id in list(state.get("obj_ids", [])):
        obj_id_i = int(obj_id)
        if obj_id_i in visible_now:
            continue
        last_seen = int(last_visible_frame_by_obj.get(obj_id_i, 0))
        if int(current_frame_idx) - last_seen >= after:
            max_area_i = int(max_visible_area_by_obj.get(obj_id_i, 0))
            if protect_area > 0 and max_area_i >= protect_area:
                expired_protected_candidates.append((max_area_i, obj_id_i))
            else:
                expired_unprotected.append(obj_id_i)

    protect_limit = int(protect_max_objects)
    expired_protected_candidates.sort(key=lambda row: (-int(row[0]), int(row[1])))
    if protect_limit > 0:
        protected = [int(obj_id) for _, obj_id in expired_protected_candidates[:protect_limit]]
        overflow = [int(obj_id) for _, obj_id in expired_protected_candidates[protect_limit:]]
    else:
        protected = [int(obj_id) for _, obj_id in expired_protected_candidates]
        overflow = []

    for obj_id_i in [*expired_unprotected, *overflow]:
        if object_removal_would_clear_conditioning(state, obj_id_i):
            continue
        predictor.remove_object(state, obj_id_i, strict=False, need_output=False)
        last_visible_frame_by_obj.pop(obj_id_i, None)
        max_visible_area_by_obj.pop(obj_id_i, None)
        pruned.append(obj_id_i)
    return pruned, protected


def object_removal_would_clear_conditioning(state: dict[str, Any], obj_id: int) -> bool:
    obj_idx = state.get("obj_id_to_idx", {}).get(int(obj_id), None)
    if obj_idx is None:
        return False
    cond_frames = set(int(v) for v in state.get("output_dict", {}).get("cond_frame_outputs", {}).keys())
    if not cond_frames:
        return True
    other_input_frames: set[int] = set()
    for other_idx, per_frame in state.get("point_inputs_per_obj", {}).items():
        if int(other_idx) != int(obj_idx):
            other_input_frames.update(int(v) for v in per_frame.keys())
    for other_idx, per_frame in state.get("mask_inputs_per_obj", {}).items():
        if int(other_idx) != int(obj_idx):
            other_input_frames.update(int(v) for v in per_frame.keys())
    return not bool(cond_frames & other_input_frames)


def prune_stream_oversized_visible_objects(
    predictor: Any,
    state: dict[str, Any] | None,
    *,
    visible_ids: np.ndarray,
    visible_masks: np.ndarray,
    max_visible_area: int,
    last_visible_frame_by_obj: dict[int, int],
    action: str = "prune",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Remove objects whose propagated mask has grown into a huge plane."""
    max_area = int(max_visible_area)
    if max_area <= 0 or visible_ids.size == 0 or visible_masks.size == 0:
        return visible_ids, visible_masks, []
    action = str(action or "prune").strip().lower()
    if action not in {"prune", "alert_only", "suppress_output"}:
        raise ValueError(f"unsupported stream oversized action: {action}")

    areas = np.asarray([int(np.count_nonzero(mask)) for mask in visible_masks.astype(bool)], dtype=np.int64)
    oversized = areas > max_area
    if not bool(np.any(oversized)):
        return visible_ids, visible_masks, []

    state_ids = set(map(int, state.get("obj_ids", []))) if state is not None else set()
    events: list[dict[str, Any]] = []
    for obj_id, area in zip(visible_ids[oversized].tolist(), areas[oversized].tolist(), strict=False):
        obj_id_i = int(obj_id)
        removed = False
        if action == "prune" and state is not None and hasattr(predictor, "remove_object") and obj_id_i in state_ids:
            if not object_removal_would_clear_conditioning(state, obj_id_i):
                predictor.remove_object(state, obj_id_i, strict=False, need_output=False)
                removed = True
        if action == "prune":
            last_visible_frame_by_obj.pop(obj_id_i, None)
        events.append(
            {
                "object_id": obj_id_i,
                "visible_area": int(area),
                "max_visible_area": int(max_area),
                "action": action,
                "object_removed": bool(removed),
                "output_suppressed": bool(action in {"prune", "suppress_output"}),
            }
        )

    keep = np.ones_like(oversized, dtype=bool) if action == "alert_only" else ~oversized
    return visible_ids[keep], visible_masks[keep], events


def disjoin_small_first_keep_ids(
    masks: np.ndarray,
    h: int,
    w: int,
    empty_ratio: float,
    *,
    claim_dropped: bool = True,
    min_area_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoin active masks with smaller objects claiming pixels first.

    The returned mask array stays aligned with the original object-id order, so
    stable SAM2 ids are preserved while large planes stop swallowing medium
    objects during current-frame ownership resolution.
    """
    if masks.size == 0:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    norm_masks: list[np.ndarray] = []
    areas: list[int] = []
    for mask in masks.astype(bool):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        norm_masks.append(mask)
        areas.append(int(np.count_nonzero(mask)))
    claimed = np.zeros((h, w), dtype=bool)
    out = [np.zeros((h, w), dtype=bool) for _ in norm_masks]
    keep = np.zeros((len(norm_masks),), dtype=bool)
    min_pixels = max(0, int(min_area_px))
    if min_pixels <= 0:
        min_pixels = int(h * w * float(empty_ratio))
    for idx in np.argsort(np.asarray(areas, dtype=np.int64)):
        mask = norm_masks[int(idx)]
        residual = mask & ~claimed
        out[int(idx)] = residual
        residual_area = int(np.count_nonzero(residual))
        keep[int(idx)] = residual_area > min_pixels
        if bool(keep[int(idx)]) or bool(claim_dropped):
            claimed |= mask
    return np.stack(out, axis=0).astype(bool), keep


def disjoin_recent_overlap_first_keep_ids(
    masks: np.ndarray,
    obj_ids: np.ndarray,
    h: int,
    w: int,
    empty_ratio: float,
    *,
    recent_mask_by_obj: dict[int, np.ndarray],
    recent_min_iou: float = 0.0,
    recent_max_area_growth: float = 0.0,
    claim_dropped: bool = True,
    min_area_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoin propagated masks while preferring same-id recent ownership.

    SAM2 can emit several active object slots over the same physical surface.
    Pure area ordering makes the visible owner flicker. This resolver first
    promotes slots whose current mask still overlaps their own recent output,
    then keeps the small-first behavior inside the stable/unstable groups.
    """
    if masks.size == 0:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if obj_ids.size != masks.shape[0]:
        raise ValueError("obj_ids must stay aligned with masks for recent-overlap stream disjoin")

    norm_masks: list[np.ndarray] = []
    areas: list[int] = []
    recent_ious: list[float] = []
    min_iou = max(0.0, float(recent_min_iou))
    max_growth = max(0.0, float(recent_max_area_growth))
    for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area_i = int(np.count_nonzero(mask))
        score = 0.0
        recent = recent_mask_by_obj.get(int(obj_id))
        if recent is not None:
            recent_bool = recent.astype(bool, copy=False)
            if recent_bool.shape[:2] != (h, w):
                recent_bool = cv2.resize(recent_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            recent_area = int(np.count_nonzero(recent_bool))
            if area_i > 0 and recent_area > 0:
                if max_growth <= 0.0 or float(area_i) <= float(recent_area) * max_growth:
                    inter = int(np.count_nonzero(mask & recent_bool))
                    union = int(area_i + recent_area - inter)
                    if union > 0:
                        score = float(inter) / float(union)
        norm_masks.append(mask)
        areas.append(area_i)
        recent_ious.append(score)

    order = sorted(
        range(len(norm_masks)),
        key=lambda idx: (
            0 if float(recent_ious[idx]) >= min_iou else 1,
            int(areas[idx]),
            -float(recent_ious[idx]),
            int(obj_ids[idx]),
        ),
    )
    claimed = np.zeros((h, w), dtype=bool)
    out = [np.zeros((h, w), dtype=bool) for _ in norm_masks]
    keep = np.zeros((len(norm_masks),), dtype=bool)
    min_pixels = max(0, int(min_area_px))
    if min_pixels <= 0:
        min_pixels = int(h * w * float(empty_ratio))
    for idx in order:
        mask = norm_masks[int(idx)]
        residual = mask & ~claimed
        out[int(idx)] = residual
        residual_area = int(np.count_nonzero(residual))
        keep[int(idx)] = residual_area > min_pixels
        if bool(keep[int(idx)]) or bool(claim_dropped):
            claimed |= mask
    return np.stack(out, axis=0).astype(bool), keep


def disjoin_small_first_recent_tiebreak_keep_ids(
    masks: np.ndarray,
    obj_ids: np.ndarray,
    h: int,
    w: int,
    empty_ratio: float,
    *,
    recent_mask_by_obj: dict[int, np.ndarray],
    recent_min_iou: float = 0.0,
    recent_max_area_growth: float = 0.0,
    claim_dropped: bool = True,
    min_area_px: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoin active masks with small objects first and recent-ID tie breaks.

    This keeps the important property from ``small_first``: chair legs, table
    legs, and other smaller structures claim pixels before large planes. Recent
    same-id overlap is only used within coarse area buckets, so stable large
    planes cannot swallow nearby smaller objects.
    """
    if masks.size == 0:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if obj_ids.size != masks.shape[0]:
        raise ValueError("obj_ids must stay aligned with masks for small-first recent-tiebreak disjoin")

    norm_masks: list[np.ndarray] = []
    areas: list[int] = []
    recent_ious: list[float] = []
    min_iou = max(0.0, float(recent_min_iou))
    max_growth = max(0.0, float(recent_max_area_growth))
    for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area_i = int(np.count_nonzero(mask))
        score = 0.0
        recent = recent_mask_by_obj.get(int(obj_id))
        if recent is not None:
            recent_bool = recent.astype(bool, copy=False)
            if recent_bool.shape[:2] != (h, w):
                recent_bool = cv2.resize(recent_bool.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            recent_area = int(np.count_nonzero(recent_bool))
            if area_i > 0 and recent_area > 0:
                if max_growth <= 0.0 or float(area_i) <= float(recent_area) * max_growth:
                    inter = int(np.count_nonzero(mask & recent_bool))
                    union = int(area_i + recent_area - inter)
                    if union > 0:
                        score = float(inter) / float(union)
        norm_masks.append(mask)
        areas.append(area_i)
        recent_ious.append(score)

    def area_bucket(area: int) -> int:
        area_i = int(area)
        if area_i < 8192:
            return area_i // 512
        if area_i < 50000:
            return 16 + ((area_i - 8192) // 2048)
        if area_i < 200000:
            return 37 + ((area_i - 50000) // 8192)
        return 56 + ((area_i - 200000) // 32768)

    order = sorted(
        range(len(norm_masks)),
        key=lambda idx: (
            area_bucket(int(areas[idx])),
            0 if float(recent_ious[idx]) >= min_iou else 1,
            int(areas[idx]),
            -float(recent_ious[idx]),
            int(obj_ids[idx]),
        ),
    )
    claimed = np.zeros((h, w), dtype=bool)
    out = [np.zeros((h, w), dtype=bool) for _ in norm_masks]
    keep = np.zeros((len(norm_masks),), dtype=bool)
    min_pixels = max(0, int(min_area_px))
    if min_pixels <= 0:
        min_pixels = int(h * w * float(empty_ratio))
    for idx in order:
        mask = norm_masks[int(idx)]
        residual = mask & ~claimed
        out[int(idx)] = residual
        residual_area = int(np.count_nonzero(residual))
        keep[int(idx)] = residual_area > min_pixels
        if bool(keep[int(idx)]) or bool(claim_dropped):
            claimed |= mask
    return np.stack(out, axis=0).astype(bool), keep


def disjoin_stream_masks(
    masks: np.ndarray,
    h: int,
    w: int,
    *,
    empty_ratio: float,
    policy: str,
    claim_dropped: bool = True,
    min_area_px: int = 0,
    obj_ids: np.ndarray | None = None,
    recent_mask_by_obj: dict[int, np.ndarray] | None = None,
    recent_min_iou: float = 0.0,
    recent_max_area_growth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    policy = str(policy or "keep_order").strip().lower()
    if policy in {"keep_order", "sam2_order"}:
        return disjoin_keep_order(masks, h, w, empty_ratio=float(empty_ratio))
    if policy in {"small_first", "smallest_first", "area_ascending"}:
        return disjoin_small_first_keep_ids(
            masks,
            h,
            w,
            empty_ratio=float(empty_ratio),
            claim_dropped=bool(claim_dropped),
            min_area_px=int(min_area_px),
        )
    if policy in {"recent_overlap_first", "recent_iou_first", "stable_recent_first"}:
        if obj_ids is None:
            raise ValueError("recent-overlap stream disjoin requires obj_ids")
        return disjoin_recent_overlap_first_keep_ids(
            masks,
            np.asarray(obj_ids, dtype=np.int64),
            h,
            w,
            empty_ratio=float(empty_ratio),
            recent_mask_by_obj=recent_mask_by_obj or {},
            recent_min_iou=float(recent_min_iou),
            recent_max_area_growth=float(recent_max_area_growth),
            claim_dropped=bool(claim_dropped),
            min_area_px=int(min_area_px),
        )
    if policy in {"small_first_recent_tiebreak", "small_recent_tiebreak", "area_first_recent_tiebreak"}:
        if obj_ids is None:
            raise ValueError("small-first recent-tiebreak stream disjoin requires obj_ids")
        return disjoin_small_first_recent_tiebreak_keep_ids(
            masks,
            np.asarray(obj_ids, dtype=np.int64),
            h,
            w,
            empty_ratio=float(empty_ratio),
            recent_mask_by_obj=recent_mask_by_obj or {},
            recent_min_iou=float(recent_min_iou),
            recent_max_area_growth=float(recent_max_area_growth),
            claim_dropped=bool(claim_dropped),
            min_area_px=int(min_area_px),
        )
    raise ValueError(f"unsupported stream disjoin policy: {policy}")


def dump_birth_masks(
    dump_root: Path | None,
    records: list[dict[str, Any]],
    *,
    scene_id: str,
    chunk_frame_index: int,
    frame_id: int,
    source: str,
    obj_ids: np.ndarray,
    masks: np.ndarray,
) -> None:
    if dump_root is None or masks.size == 0:
        return
    mask_dir = dump_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
        path = mask_dir / f"frame_{int(frame_id):06d}_obj_{int(obj_id):06d}_{source}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        records.append(
            {
                "scene_id": str(scene_id),
                "chunk_frame_index": int(chunk_frame_index),
                "frame_id": int(frame_id),
                "obj_id": int(obj_id),
                "source": str(source),
                "mask_path": str(path),
                "mask_area": int(np.count_nonzero(mask)),
            }
        )


def run(args: SimpleNamespace) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = (REPO_ROOT / args.rgb_root).resolve() if not Path(args.rgb_root).is_absolute() else Path(args.rgb_root)
    rgb_root = rgb_root / args.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])

    variant = str(args.variant_id)
    output_base = Path(args.output_root)
    if not output_base.is_absolute():
        output_base = REPO_ROOT / output_base
    output_root = output_base / variant
    label_dir = output_root / "labels"
    overlay_dir = output_root / "overlays"
    sheet_dir = output_root / "sheets"
    for directory in (label_dir, overlay_dir, sheet_dir, output_root / "videos"):
        directory.mkdir(parents=True, exist_ok=True)
    birth_dump_dir = Path(str(args.birth_dump_dir)) if str(args.birth_dump_dir).strip() else None
    if birth_dump_dir is not None and not birth_dump_dir.is_absolute():
        birth_dump_dir = REPO_ROOT / birth_dump_dir
    birth_records: list[dict[str, Any]] = []
    if birth_dump_dir is not None:
        birth_dump_dir.mkdir(parents=True, exist_ok=True)
    video_dir = make_numeric_frame_dir(frame_paths, output_root)
    rgbs = [read_rgb(path) for path in frame_paths]
    h, w = rgbs[0].shape[:2]

    t_setup = time.time()
    models = setup_models(args)
    setup_sec = time.time() - t_setup
    segmentor = models["segmentor"]
    tracker_model = models["tracker_model"]

    records: list[dict[str, Any]] = []
    frame_diagnostics: list[dict[str, Any]] = []
    per_frame_masks: list[np.ndarray] = [np.zeros((0, h, w), dtype=bool) for _ in frame_ids]
    per_frame_ids: list[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in frame_ids]
    to_prop_masks: list[np.ndarray] = [np.zeros((0, h, w), dtype=bool) for _ in frame_ids]
    to_prop_ids: list[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in frame_ids]
    acc_masks: list[list[np.ndarray]] = [[] for _ in frame_ids]
    acc_ids: list[list[int]] = [[] for _ in frame_ids]

    total_t0 = time.time()
    s1_seed = stable_seed(args.seed, args.scene_id, frame_ids[0], args.stage1_num_pts, args.stage1_point_mode, "baseline-x-stage1-largest")
    s1_points, s1_point_meta = make_points_yx_torch(int(args.stage1_num_pts), s1_seed, str(args.stage1_point_mode))
    t0 = time.time()
    stage1_masks, stage1_stats = run_sam2_point_segment_choice(
        segmentor,
        rgbs[0],
        points_yx=s1_points,
        region_mask=None,
        points_per_batch=int(args.points_per_batch),
        choice_policy=str(args.stage1_choice_policy),
        iou_threshold=float(args.stage1_iou_threshold),
        stability_threshold=float(args.stage1_stability_threshold),
        stability_score_offset=float(args.stability_score_offset),
        model_mask_thresh=float(args.model_mask_thresh),
        box_nms_thresh=float(args.box_nms_thresh),
        empty_ratio=float(args.empty_ratio),
        apply_box_nms=bool(args.stage1_apply_box_nms),
        nms_score_type=str(args.stage1_nms_score_type),
    )
    stage1_sec = time.time() - t0

    uncovered0 = uncovered_from_masks(stage1_masks, h, w)
    s2_seed = stable_seed(args.seed, args.scene_id, frame_ids[0], args.stage2_num_pts, "baseline-x-stage2-smallest")
    s2_points = sample_points_from_mask_yx(uncovered0, int(args.stage2_num_pts), s2_seed, inner_margin=int(args.gap_inner_margin))
    t0 = time.time()
    if int(s2_points.shape[0]) > 0:
        stage2_masks, stage2_stats = run_sam2_point_segment_choice(
            segmentor,
            rgbs[0],
            points_yx=s2_points,
            region_mask=uncovered0,
            points_per_batch=int(args.points_per_batch),
            choice_policy=str(args.stage2_choice_policy),
            iou_threshold=float(args.stage2_iou_threshold),
            stability_threshold=float(args.stage2_stability_threshold),
            stability_score_offset=float(args.stability_score_offset),
            model_mask_thresh=float(args.model_mask_thresh),
            box_nms_thresh=float(args.box_nms_thresh),
            empty_ratio=float(args.empty_ratio),
            apply_box_nms=bool(args.stage2_apply_box_nms),
            nms_score_type="stability",
        )
    else:
        stage2_masks = np.zeros((0, h, w), dtype=bool)
        stage2_stats = {
            "choice_policy": str(args.stage2_choice_policy),
            "raw_multimask_option_count": 0,
            "prompt_with_good_mask_count": 0,
            "pre_nms_mask_count": 0,
            "post_disjoint_mask_count": 0,
            "apply_box_nms": bool(args.stage2_apply_box_nms),
            "nms_score_type": "stability",
        }
    stage2_sec = time.time() - t0

    if stage2_masks.size:
        masks0 = np.concatenate([stage1_masks, stage2_masks], axis=0)
    else:
        masks0 = stage1_masks
    next_obj_id = int(masks0.shape[0])
    per_frame_masks[0] = masks0
    per_frame_ids[0] = np.arange(next_obj_id, dtype=np.int64)
    to_prop_masks[0] = masks0
    to_prop_ids[0] = per_frame_ids[0]
    dump_birth_masks(
        birth_dump_dir,
        birth_records,
        scene_id=str(args.scene_id),
        chunk_frame_index=0,
        frame_id=int(frame_ids[0]),
        source="frame0_seed",
        obj_ids=per_frame_ids[0],
        masks=masks0,
    )
    frame_diagnostics.append(
        {
            "chunk_frame_index": 0,
            "frame_id": int(frame_ids[0]),
            "frame0_stage1_runtime_sec": float(stage1_sec),
            "frame0_stage2_runtime_sec": float(stage2_sec),
            "frame0_stage1_mask_count": int(stage1_masks.shape[0]),
            "frame0_stage2_mask_count": int(stage2_masks.shape[0]),
            "frame0_initial_mask_count": int(masks0.shape[0]),
            "frame0_uncovered_ratio_after_stage1": float(np.count_nonzero(uncovered0)) / float(uncovered0.size),
            "stage1_stats": stage1_stats,
            "stage2_stats": stage2_stats,
        }
    )

    total_tracking_sec = 0.0
    total_gap_seg_sec = 0.0
    empty_propagation_frames = 0
    propagation_mode = str(args.propagation_mode)
    stream_disjoin_policy = str(getattr(args, "stream_disjoin_policy", "keep_order"))
    stream_disjoin_claim_dropped = bool(getattr(args, "stream_disjoin_claim_dropped", True))
    stream_disjoin_min_area_px = max(0, int(getattr(args, "stream_disjoin_min_area_px", 0)))
    stream_disjoin_recent_min_iou = max(0.0, float(getattr(args, "stream_disjoin_recent_min_iou", 0.0)))
    stream_disjoin_recent_max_area_growth = max(
        0.0,
        float(getattr(args, "stream_disjoin_recent_max_area_growth", 0.0)),
    )
    stream_oversized_prune_action = str(getattr(args, "stream_oversized_prune_action", "prune"))
    stream_state: dict[str, Any] | None = None
    stream_memory_prune_events: list[dict[str, Any]] = []
    stream_object_prune_events: list[dict[str, Any]] = []
    stream_protected_invisible_events: list[dict[str, Any]] = []
    stream_min_visible_area = int(getattr(args, "stream_prune_min_visible_area", 0))
    stream_prune_protect_min_ever_area = int(getattr(args, "stream_prune_protect_min_ever_area", 0))
    stream_prune_protect_max_objects = int(getattr(args, "stream_prune_protect_max_objects", 0))
    stream_max_visible_area_arg = int(getattr(args, "stream_prune_max_visible_area", 0))
    stream_max_visible_area_ratio = float(getattr(args, "stream_prune_max_visible_area_ratio", 0.0))
    stream_max_visible_area = stream_max_visible_area_arg
    if stream_max_visible_area <= 0 and stream_max_visible_area_ratio > 0.0:
        stream_max_visible_area = int(round(float(h * w) * stream_max_visible_area_ratio))
    last_visible_frame_by_obj: dict[int, int] = {}
    max_visible_area_by_obj: dict[int, int] = {}
    recent_output_mask_by_obj: dict[int, np.ndarray] = {}
    recent_output_frame_by_obj: dict[int, int] = {}
    for obj_id, mask in zip(per_frame_ids[0].tolist(), per_frame_masks[0].astype(bool), strict=False):
        area_i = int(np.count_nonzero(mask))
        max_visible_area_by_obj[int(obj_id)] = area_i
        if area_i >= max(0, stream_min_visible_area):
            last_visible_frame_by_obj[int(obj_id)] = 0
            recent_output_mask_by_obj[int(obj_id)] = mask.astype(bool, copy=True)
            recent_output_frame_by_obj[int(obj_id)] = 0
    if propagation_mode == "streaming_state":
        import torch

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream_state = tracker_model.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=False,
            )
        add_masks_to_stream_state(
            tracker_model,
            stream_state,
            tracker=str(args.tracker_backend),
            frame_idx=0,
            obj_ids=to_prop_ids[0],
            masks=to_prop_masks[0],
        )
    elif propagation_mode != "reseed_full_video":
        raise ValueError(f"unsupported propagation_mode: {propagation_mode}")

    for t in range(len(frame_ids) - 1):
        prop_t0 = time.time()
        if stream_state is not None:
            current_ids, current_masks_pre = infer_stream_frame(
                tracker_model,
                stream_state,
                frame_idx=t + 1,
            )
            prop_sec = time.time() - prop_t0
            total_tracking_sec += prop_sec
            if current_masks_pre.size:
                current_masks_all, keep = disjoin_stream_masks(
                    current_masks_pre,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                    policy=stream_disjoin_policy,
                    claim_dropped=stream_disjoin_claim_dropped,
                    min_area_px=stream_disjoin_min_area_px,
                    obj_ids=current_ids,
                    recent_mask_by_obj=recent_output_mask_by_obj,
                    recent_min_iou=stream_disjoin_recent_min_iou,
                    recent_max_area_growth=stream_disjoin_recent_max_area_growth,
                )
                current_masks = current_masks_all[keep]
                current_ids = current_ids[keep]
            else:
                empty_propagation_frames += 1
                current_ids = np.zeros((0,), dtype=np.int64)
                current_masks = np.zeros((0, h, w), dtype=bool)
                current_masks_pre = np.zeros((0, h, w), dtype=bool)
        else:
            propagated = propagate_new_masks_chunked(
                tracker_model,
                tracker=str(args.tracker_backend),
                video_dir=video_dir,
                seed_frame=t,
                obj_ids=to_prop_ids[t],
                masks=to_prop_masks[t],
                total_frames=len(frame_ids),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                chunk_size=int(args.propagation_chunk_size),
            )
            prop_sec = time.time() - prop_t0
            total_tracking_sec += prop_sec
            for future in range(t + 1, len(frame_ids)):
                for obj_id, mask in propagated.get(future, {}).items():
                    acc_ids[future].append(int(obj_id))
                    acc_masks[future].append(mask.astype(bool))

            if acc_masks[t + 1]:
                current_ids = np.asarray(acc_ids[t + 1], dtype=np.int64)
                current_masks_pre = np.stack(acc_masks[t + 1], axis=0).astype(bool)
                current_masks_all, keep = disjoin_stream_masks(
                    current_masks_pre,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                    policy=stream_disjoin_policy,
                    claim_dropped=stream_disjoin_claim_dropped,
                    min_area_px=stream_disjoin_min_area_px,
                    obj_ids=current_ids,
                    recent_mask_by_obj=recent_output_mask_by_obj,
                    recent_min_iou=stream_disjoin_recent_min_iou,
                    recent_max_area_growth=stream_disjoin_recent_max_area_growth,
                )
                current_masks = current_masks_all[keep]
                current_ids = current_ids[keep]
            else:
                empty_propagation_frames += 1
                current_ids = np.zeros((0,), dtype=np.int64)
                current_masks = np.zeros((0, h, w), dtype=bool)
                current_masks_pre = np.zeros((0, h, w), dtype=bool)

        if stream_state is not None:
            stream_state["v106_current_frame_idx_for_birth_filter"] = int(t + 1)
            stream_state["v106_current_visible_count_for_birth_filter"] = int(current_ids.shape[0])
            if current_masks.size:
                current_union = np.any(current_masks.astype(bool, copy=False), axis=0)
                stream_state["v106_current_foreground_ratio_for_birth_filter"] = float(
                    np.count_nonzero(current_union)
                ) / float(current_union.size)
            else:
                stream_state["v106_current_foreground_ratio_for_birth_filter"] = 0.0

        uncovered = uncovered_from_masks(current_masks, h, w)
        gap_seed = stable_seed(args.seed, args.scene_id, frame_ids[t + 1], args.gap_num_pts_active, "baseline-x-gap-smallest")
        if str(args.gap_sampler) == "component_adaptive":
            gap_points, gap_sampling_meta = sample_component_adaptive_points_yx(
                uncovered,
                max_points=int(args.gap_max_points),
                min_component_area=int(args.gap_min_component_area),
                base_points_per_component=int(args.gap_base_points_per_component),
                area_per_extra_point=int(args.gap_area_per_extra_point),
                max_points_per_component=int(args.gap_max_points_per_component),
                min_image_edge_distance_px=int(args.gap_min_image_edge_distance_px),
                seed=gap_seed,
            )
        elif str(args.gap_sampler) == "uniform_random":
            gap_points = sample_points_from_mask_yx(
                uncovered,
                int(args.gap_num_pts_active),
                gap_seed,
                inner_margin=int(args.gap_inner_margin),
            )
            gap_sampling_meta = {
                "sampler": "uniform_random",
                "point_count": int(gap_points.shape[0]),
                "max_points": int(args.gap_num_pts_active),
            }
        else:
            raise ValueError(f"unsupported gap sampler: {args.gap_sampler}")
        gap_t0 = time.time()
        if int(gap_points.shape[0]) > 0:
            gap_masks, gap_stats = run_sam2_point_segment_choice(
                segmentor,
                rgbs[t + 1],
                points_yx=gap_points,
                region_mask=uncovered,
                points_per_batch=int(args.points_per_batch),
                choice_policy=str(args.gap_choice_policy),
                iou_threshold=float(args.gap_iou_threshold),
                stability_threshold=float(args.gap_stability_threshold),
                stability_score_offset=float(args.stability_score_offset),
                model_mask_thresh=float(args.model_mask_thresh),
                box_nms_thresh=float(args.box_nms_thresh),
                empty_ratio=float(args.empty_ratio),
                apply_box_nms=bool(args.gap_apply_box_nms),
                nms_score_type="stability",
                relaxed_min_region_ratio=float(
                    getattr(args, "gap_relaxed_min_uncovered_ratio", 0.0)
                ),
                relaxed_iou_threshold=float(getattr(args, "gap_relaxed_iou_threshold", 0.0)),
                relaxed_stability_threshold=float(
                    getattr(args, "gap_relaxed_stability_threshold", 0.0)
                ),
                relaxed_min_clipped_area=int(getattr(args, "gap_relaxed_min_clipped_area", 0)),
                small_mask_max_area=int(getattr(args, "gap_small_mask_max_area", 0)),
                small_mask_min_pred_iou=float(getattr(args, "gap_small_mask_min_pred_iou", 0.0)),
            )
        else:
            gap_masks = np.zeros((0, h, w), dtype=bool)
            gap_stats = {
                "choice_policy": str(args.gap_choice_policy),
                "raw_multimask_option_count": 0,
                "prompt_with_good_mask_count": 0,
                "pre_nms_mask_count": 0,
                "post_disjoint_mask_count": 0,
                "apply_box_nms": bool(args.gap_apply_box_nms),
                "nms_score_type": "stability",
                "small_mask_quality_filter": {
                    "enabled": bool(
                        int(getattr(args, "gap_small_mask_max_area", 0)) > 0
                        and float(getattr(args, "gap_small_mask_min_pred_iou", 0.0)) > 0.0
                    ),
                    "small_mask_max_area": int(getattr(args, "gap_small_mask_max_area", 0)),
                    "small_mask_min_pred_iou": float(getattr(args, "gap_small_mask_min_pred_iou", 0.0)),
                    "dropped_mask_count": 0,
                    "dropped_mask_records_sample": [],
                },
            }
        gap_sec = time.time() - gap_t0
        total_gap_seg_sec += gap_sec
        gap_records = list((gap_stats.get("post_disjoint_selected_prompt_records") or []))
        gap_masks, gap_records, gap_output_quality_filter = filter_gap_output_masks_by_quality(
            gap_masks,
            gap_records,
            min_pred_iou=float(getattr(args, "gap_output_min_pred_iou", 0.0)),
            min_stability=float(getattr(args, "gap_output_min_stability", 0.0)),
            allow_relaxed=bool(getattr(args, "gap_output_allow_relaxed", True)),
        )
        gap_stats["output_quality_filter"] = gap_output_quality_filter
        gap_ids, next_obj_id, gap_reuse_events = assign_gap_ids_with_recent_reuse(
            gap_masks,
            current_ids=current_ids,
            next_obj_id=int(next_obj_id),
            recent_mask_by_obj=recent_output_mask_by_obj,
            recent_frame_by_obj=recent_output_frame_by_obj,
            current_frame_idx=int(t + 1),
            reuse_window_frames=int(getattr(args, "gap_reuse_recent_id_window", 0)),
            reuse_iou_threshold=float(getattr(args, "gap_reuse_recent_id_iou", 0.0)),
            reuse_min_area=int(getattr(args, "gap_reuse_recent_id_min_area", 0)),
            large_reuse_iou_threshold=float(getattr(args, "gap_large_reuse_recent_id_iou", 0.0)),
            large_reuse_min_area=int(getattr(args, "gap_large_reuse_min_area", 0)),
            large_reuse_max_area_ratio=float(getattr(args, "gap_large_reuse_max_area_ratio", 0.0)),
        )
        gap_admit_ids, gap_admit_masks, gap_admission_stats = select_gap_admission_masks(
            gap_ids,
            gap_masks,
            gap_records=gap_records,
            uncovered_mask=uncovered,
            ownership_mask_by_obj=recent_output_mask_by_obj,
            ownership_frame_by_obj=recent_output_frame_by_obj,
            enabled=bool(getattr(args, "gap_delayed_admission_enabled", False)),
            frame_idx=int(t + 1),
            frame_id=int(frame_ids[t + 1]),
            min_area=int(getattr(args, "birth_admission_min_area", 0)),
            max_area=int(getattr(args, "birth_admission_max_area", 0)),
            max_uncovered_ratio=float(getattr(args, "birth_admission_max_uncovered_ratio", 0.0)),
            max_bbox_frac=float(getattr(args, "birth_admission_max_bbox_frac", 0.0)),
            max_edge_touch_count=int(getattr(args, "birth_admission_max_edge_touch_count", -1)),
            min_extent=float(getattr(args, "birth_admission_min_extent", 0.0)),
            min_core_area=int(getattr(args, "birth_admission_min_core_area_px", 0)),
            shape_min_uncovered_ratio=float(getattr(args, "birth_admission_shape_min_uncovered_ratio", 0.0)),
            max_per_frame=int(getattr(args, "birth_admission_max_per_frame", 0)),
            min_pred_iou=float(getattr(args, "gap_admission_min_pred_iou", 0.0)),
            min_stability=float(getattr(args, "gap_admission_min_stability", 0.0)),
            allow_relaxed=bool(getattr(args, "gap_admission_allow_relaxed", True)),
            anti_merge_core_window_frames=int(getattr(args, "gap_anti_merge_core_window_frames", 0)),
            anti_merge_core_erode_px=int(getattr(args, "gap_anti_merge_core_erode_px", 0)),
            anti_merge_core_min_area=int(getattr(args, "gap_anti_merge_core_min_area", 0)),
            anti_merge_core_min_overlap_px=int(getattr(args, "gap_anti_merge_core_min_overlap_px", 0)),
            anti_merge_core_min_overlap_ratio=float(
                getattr(args, "gap_anti_merge_core_min_overlap_ratio", 0.0)
            ),
            anti_merge_max_overlap_objects=int(getattr(args, "gap_anti_merge_max_overlap_objects", 1)),
        )
        dump_birth_masks(
            birth_dump_dir,
            birth_records,
            scene_id=str(args.scene_id),
            chunk_frame_index=int(t + 1),
            frame_id=int(frame_ids[t + 1]),
            source="gap_birth",
            obj_ids=gap_ids,
            masks=gap_masks,
        )
        to_prop_masks[t + 1] = gap_admit_masks
        to_prop_ids[t + 1] = gap_admit_ids
        if stream_state is not None and gap_admit_masks.size:
            add_masks_to_stream_state(
                tracker_model,
                stream_state,
                tracker=str(args.tracker_backend),
                frame_idx=t + 1,
                obj_ids=gap_admit_ids,
                masks=gap_admit_masks,
            )
        if gap_masks.size:
            per_frame_masks[t + 1] = np.concatenate([current_masks, gap_masks], axis=0)
            per_frame_ids[t + 1] = np.concatenate([current_ids, gap_ids], axis=0)
        else:
            per_frame_masks[t + 1] = current_masks
            per_frame_ids[t + 1] = current_ids
        pruned_object_ids: list[int] = []
        protected_invisible_object_ids: list[int] = []
        oversized_prune_events: list[dict[str, Any]] = []
        pruned_noncond_frames: list[int] = []
        if stream_max_visible_area > 0:
            (
                per_frame_ids[t + 1],
                per_frame_masks[t + 1],
                oversized_prune_events,
            ) = prune_stream_oversized_visible_objects(
                tracker_model,
                stream_state,
                visible_ids=per_frame_ids[t + 1],
                visible_masks=per_frame_masks[t + 1],
                max_visible_area=stream_max_visible_area,
                last_visible_frame_by_obj=last_visible_frame_by_obj,
                action=stream_oversized_prune_action,
            )
        if stream_state is not None:
            pruned_object_ids, protected_invisible_object_ids = prune_stream_invisible_objects(
                tracker_model,
                stream_state,
                current_frame_idx=int(t + 1),
                visible_ids=per_frame_ids[t + 1],
                visible_masks=per_frame_masks[t + 1],
                last_visible_frame_by_obj=last_visible_frame_by_obj,
                max_visible_area_by_obj=max_visible_area_by_obj,
                min_visible_area=stream_min_visible_area,
                invisible_after_frames=int(getattr(args, "stream_prune_invisible_after_frames", 0)),
                protect_min_ever_area=stream_prune_protect_min_ever_area,
                protect_max_objects=stream_prune_protect_max_objects,
            )
            pruned_noncond_frames = prune_stream_noncond_memory(
                stream_state,
                current_frame_idx=int(t + 1),
                keep_recent_frames=int(getattr(args, "stream_keep_noncond_frames", 0)),
            )
            if pruned_object_ids:
                stream_object_prune_events.append(
                    {
                        "chunk_frame_index": int(t + 1),
                        "frame_id": int(frame_ids[t + 1]),
                        "pruned_object_ids": [int(v) for v in pruned_object_ids],
                        "active_stream_object_count_after_prune": int(len(stream_state.get("obj_ids", []))),
                    }
                )
            if protected_invisible_object_ids:
                stream_protected_invisible_events.append(
                    {
                        "chunk_frame_index": int(t + 1),
                        "frame_id": int(frame_ids[t + 1]),
                        "protect_min_ever_area": int(stream_prune_protect_min_ever_area),
                        "protect_max_objects": int(stream_prune_protect_max_objects),
                        "protected_object_ids": [int(v) for v in protected_invisible_object_ids],
                        "protected_object_max_visible_area": {
                            str(int(v)): int(max_visible_area_by_obj.get(int(v), 0))
                            for v in protected_invisible_object_ids[:64]
                        },
                        "active_stream_object_count_after_prune": int(len(stream_state.get("obj_ids", []))),
                    }
                )
            if oversized_prune_events:
                stream_object_prune_events.append(
                    {
                        "chunk_frame_index": int(t + 1),
                        "frame_id": int(frame_ids[t + 1]),
                        "reason": "oversized_visible_area",
                        "max_visible_area": int(stream_max_visible_area),
                        "events": oversized_prune_events,
                        "pruned_object_ids": [int(row["object_id"]) for row in oversized_prune_events],
                        "active_stream_object_count_after_prune": int(len(stream_state.get("obj_ids", []))),
                    }
                )
            if pruned_noncond_frames:
                stream_memory_prune_events.append(
                    {
                        "chunk_frame_index": int(t + 1),
                        "frame_id": int(frame_ids[t + 1]),
                        "pruned_noncond_frame_indices": [int(v) for v in pruned_noncond_frames],
                        "kept_noncond_frame_count": int(
                            len(stream_state.get("output_dict", {}).get("non_cond_frame_outputs", {}))
                        ),
                    }
                )
            empty_every = int(getattr(args, "stream_empty_cache_every", 0))
            empty_on_prune = bool(getattr(args, "stream_empty_cache_on_prune", True))
            if (
                (empty_on_prune and (pruned_object_ids or oversized_prune_events or pruned_noncond_frames))
                or (empty_every > 0 and int(t + 1) % empty_every == 0)
            ):
                torch.cuda.empty_cache()
        output_relabel_events: list[dict[str, Any]] = []
        if per_frame_ids[t + 1].size and per_frame_masks[t + 1].size:
            (
                per_frame_ids[t + 1],
                per_frame_masks[t + 1],
                output_relabel_events,
            ) = canonicalize_output_ids_with_recent_overlap(
                per_frame_ids[t + 1],
                per_frame_masks[t + 1],
                recent_mask_by_obj=recent_output_mask_by_obj,
                recent_frame_by_obj=recent_output_frame_by_obj,
                current_frame_idx=int(t + 1),
                reuse_window_frames=int(getattr(args, "output_reuse_recent_id_window", 0)),
                reuse_iou_threshold=float(getattr(args, "output_reuse_recent_id_iou", 0.0)),
                reuse_min_area=int(getattr(args, "output_reuse_recent_id_min_area", 0)),
                canonical_preference=str(getattr(args, "output_reuse_recent_id_preference", "lower_id")),
                prevent_collision_union=bool(getattr(args, "output_reuse_prevent_collision_union", False)),
                large_reuse_iou_threshold=float(getattr(args, "output_large_reuse_recent_id_iou", 0.0)),
                large_reuse_min_area=int(getattr(args, "output_large_reuse_min_area", 0)),
                large_reuse_max_area_ratio=float(getattr(args, "output_large_reuse_max_area_ratio", 0.0)),
            )
        output_fragment_events: list[dict[str, Any]] = []
        if per_frame_ids[t + 1].size and per_frame_masks[t + 1].size:
            (
                per_frame_ids[t + 1],
                per_frame_masks[t + 1],
                output_fragment_events,
            ) = merge_or_suppress_output_fragments(
                per_frame_ids[t + 1],
                per_frame_masks[t + 1],
                h,
                w,
                fragment_max_area=int(getattr(args, "output_fragment_max_area", 0)),
                suppress_max_area=int(getattr(args, "output_fragment_suppress_max_area", 0)),
                merge_dilate_px=int(getattr(args, "output_fragment_merge_dilate_px", 2)),
                merge_min_touch_px=int(getattr(args, "output_fragment_merge_min_touch_px", 16)),
                merge_min_touch_ratio=float(getattr(args, "output_fragment_merge_min_touch_ratio", 0.02)),
                merge_min_neighbor_area=int(getattr(args, "output_fragment_merge_min_neighbor_area", 20000)),
            )
        if per_frame_ids[t + 1].size and per_frame_masks[t + 1].size:
            for obj_id, mask in zip(
                per_frame_ids[t + 1].tolist(),
                per_frame_masks[t + 1].astype(bool),
                strict=False,
            ):
                if int(np.count_nonzero(mask)) >= max(0, stream_min_visible_area):
                    recent_output_mask_by_obj[int(obj_id)] = mask.astype(bool, copy=True)
                    recent_output_frame_by_obj[int(obj_id)] = int(t + 1)
        frame_diagnostics.append(
            {
                "chunk_frame_index": int(t + 1),
                "frame_id": int(frame_ids[t + 1]),
                "propagation_seed_frame_index": int(t),
                "propagation_mode": propagation_mode,
                "stream_disjoin_policy": stream_disjoin_policy,
                "stream_disjoin_claim_dropped": bool(stream_disjoin_claim_dropped),
                "stream_disjoin_min_area_px": int(stream_disjoin_min_area_px),
                "stream_disjoin_recent_min_iou": float(stream_disjoin_recent_min_iou),
                "stream_disjoin_recent_max_area_growth": float(stream_disjoin_recent_max_area_growth),
                "stream_oversized_prune_action": stream_oversized_prune_action,
                "propagation_runtime_sec": float(prop_sec),
                "new_seed_mask_count": int(to_prop_masks[t].shape[0]),
                "propagation_chunk_size": int(args.propagation_chunk_size),
                "propagated_pre_disjoin_count": int(current_masks_pre.shape[0]),
                "propagated_post_disjoin_count": int(current_masks.shape[0]),
                "propagated_disjoin_drop_count": int(current_masks_pre.shape[0] - current_masks.shape[0]),
                "gap_runtime_sec": float(gap_sec),
                "gap_mask_count": int(gap_masks.shape[0]),
                "gap_output_mask_count": int(gap_masks.shape[0]),
                "gap_admitted_to_stream_count": int(gap_admit_masks.shape[0]),
                "gap_output_only_mask_count": int(gap_admission_stats.get("output_only_mask_count", 0)),
                "gap_output_quality_drop_count": int(gap_output_quality_filter.get("dropped_mask_count", 0)),
                "gap_output_quality_filter": gap_output_quality_filter,
                "gap_admission_stats": gap_admission_stats,
                "gap_anti_merge_core_window_frames": int(
                    getattr(args, "gap_anti_merge_core_window_frames", 0)
                ),
                "gap_anti_merge_core_erode_px": int(getattr(args, "gap_anti_merge_core_erode_px", 0)),
                "gap_anti_merge_core_min_area": int(getattr(args, "gap_anti_merge_core_min_area", 0)),
                "gap_anti_merge_core_min_overlap_px": int(
                    getattr(args, "gap_anti_merge_core_min_overlap_px", 0)
                ),
                "gap_anti_merge_core_min_overlap_ratio": float(
                    getattr(args, "gap_anti_merge_core_min_overlap_ratio", 0.0)
                ),
                "gap_anti_merge_max_overlap_objects": int(
                    getattr(args, "gap_anti_merge_max_overlap_objects", 1)
                ),
                "final_frame_mask_count": int(per_frame_masks[t + 1].shape[0]),
                "stream_pruned_object_ids": [int(v) for v in pruned_object_ids],
                "stream_invisible_protected_object_ids": [int(v) for v in protected_invisible_object_ids],
                "stream_invisible_protected_objects": int(len(protected_invisible_object_ids)),
                "stream_oversized_prune_events": oversized_prune_events,
                "stream_pruned_noncond_frame_indices": [int(v) for v in pruned_noncond_frames],
                "stream_active_object_count_after_prune": int(len(stream_state.get("obj_ids", []))) if stream_state is not None else 0,
                "stream_noncond_frame_count_after_prune": int(
                    len(stream_state.get("output_dict", {}).get("non_cond_frame_outputs", {}))
                )
                if stream_state is not None
                else 0,
                "uncovered_ratio_before_gap": float(np.count_nonzero(uncovered)) / float(uncovered.size),
                "gap_sampling": gap_sampling_meta,
                "gap_stats": gap_stats,
                "gap_reuse_recent_id_window": int(getattr(args, "gap_reuse_recent_id_window", 0)),
                "gap_reuse_recent_id_iou": float(getattr(args, "gap_reuse_recent_id_iou", 0.0)),
                "gap_reuse_recent_id_min_area": int(getattr(args, "gap_reuse_recent_id_min_area", 0)),
                "gap_large_reuse_recent_id_iou": float(getattr(args, "gap_large_reuse_recent_id_iou", 0.0)),
                "gap_large_reuse_min_area": int(getattr(args, "gap_large_reuse_min_area", 0)),
                "gap_large_reuse_max_area_ratio": float(getattr(args, "gap_large_reuse_max_area_ratio", 0.0)),
                "gap_reuse_event_count": int(len(gap_reuse_events)),
                "gap_reuse_events": gap_reuse_events,
                "output_reuse_recent_id_window": int(getattr(args, "output_reuse_recent_id_window", 0)),
                "output_reuse_recent_id_iou": float(getattr(args, "output_reuse_recent_id_iou", 0.0)),
                "output_reuse_recent_id_min_area": int(getattr(args, "output_reuse_recent_id_min_area", 0)),
                "output_large_reuse_recent_id_iou": float(getattr(args, "output_large_reuse_recent_id_iou", 0.0)),
                "output_large_reuse_min_area": int(getattr(args, "output_large_reuse_min_area", 0)),
                "output_large_reuse_max_area_ratio": float(getattr(args, "output_large_reuse_max_area_ratio", 0.0)),
                "output_reuse_recent_id_preference": str(
                    getattr(args, "output_reuse_recent_id_preference", "lower_id")
                ),
                "output_relabel_event_count": int(len(output_relabel_events)),
                "output_relabel_collision_guard_event_count": int(
                    sum(str(row.get("action")) == "collision_guard_revert" for row in output_relabel_events)
                ),
                "output_relabel_events": output_relabel_events,
                "output_fragment_event_count": int(len(output_fragment_events)),
                "output_fragment_merge_event_count": int(
                    sum(str(row.get("action")) == "merge_to_neighbor" for row in output_fragment_events)
                ),
                "output_fragment_suppress_event_count": int(
                    sum(str(row.get("action")) == "suppress_to_background" for row in output_fragment_events)
                ),
                "output_fragment_events_sample": output_fragment_events[:64],
            }
        )
        print(
            json.dumps(
                {
                    "variant": variant,
                    "frame_index": int(t + 1),
                    "frame_id": int(frame_ids[t + 1]),
                    "propagated": int(current_masks.shape[0]),
                    "gap": int(gap_masks.shape[0]),
                    "gap_admitted": int(gap_admit_masks.shape[0]),
                    "gap_output_only": int(gap_admission_stats.get("output_only_mask_count", 0)),
                    "final": int(per_frame_masks[t + 1].shape[0]),
                    "prop_sec": round(prop_sec, 3),
                    "gap_sec": round(gap_sec, 3),
                    "gap_quality_dropped": int(
                        (gap_stats.get("small_mask_quality_filter") or {}).get("dropped_mask_count") or 0
                    ),
                    "gap_output_quality_dropped": int(
                        (gap_stats.get("output_quality_filter") or {}).get("dropped_mask_count") or 0
                    ),
                    "collision_guard": int(
                        sum(str(row.get("action")) == "collision_guard_revert" for row in output_relabel_events)
                    ),
                    "stream_pruned_objects": int(len(pruned_object_ids)),
                    "stream_protected_invisible_objects": int(len(protected_invisible_object_ids)),
                    "stream_pruned_noncond_frames": int(len(pruned_noncond_frames)),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    overlay_paths: list[Path] = []
    for chunk_idx, (frame_id, rgb, obj_ids, masks) in enumerate(zip(frame_ids, rgbs, per_frame_ids, per_frame_masks, strict=True)):
        label = label_from_id_masks(obj_ids, masks, h, w)
        label_path = label_dir / f"frame_{int(frame_id):06d}.png"
        cv2.imwrite(str(label_path), label)
        overlay = overlay_label(rgb, label)
        stats = mask_stats(label)
        annotated = annotate_frame(
            overlay,
            f"{variant} frame {chunk_idx:02d} / id {int(frame_id)}",
            [
                f"masks={stats['visible_id_count']} fg={stats['foreground_ratio']:.3f} ids={int(obj_ids.size)}",
                f"mode={str(args.propagation_mode)} gap={str(args.gap_sampler)} max{int(args.gap_max_points)}",
            ],
        )
        overlay_path = overlay_dir / f"frame_{chunk_idx:02d}_id_{int(frame_id):06d}.jpg"
        annotated.save(overlay_path, quality=95)
        overlay_paths.append(overlay_path)
        records.append(
            {
                "chunk_frame_index": int(chunk_idx),
                "frame_id": int(frame_id),
                "label_path": str(label_path),
                "overlay_path": str(overlay_path),
                "object_id_count": int(obj_ids.size),
                "visible_id_count": int(stats["visible_id_count"]),
                "foreground_ratio": float(stats["foreground_ratio"]),
            }
        )

    sheet_paths: list[str] = []
    for start in range(0, len(overlay_paths), 8):
        part = overlay_paths[start : start + 8]
        end = start + len(part) - 1
        sheet_path = sheet_dir / f"{variant}_{args.scene_id}_frames_{start:02d}_{end:02d}_4x2.jpg"
        make_sheet_grid(part, sheet_path, int(args.sheet_cell_width), cols=4)
        sheet_paths.append(str(sheet_path))
    video_path = output_root / "videos" / f"{variant}_{args.scene_id}_chunk0.mp4"
    write_video(overlay_paths, video_path, fps=float(args.fps))

    total_sec = time.time() - total_t0
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    gap_reuse_event_count = sum(int(row.get("gap_reuse_event_count") or 0) for row in frame_diagnostics)
    output_relabel_event_count = sum(int(row.get("output_relabel_event_count") or 0) for row in frame_diagnostics)
    output_relabel_collision_guard_event_count = sum(
        int(row.get("output_relabel_collision_guard_event_count") or 0) for row in frame_diagnostics
    )
    output_fragment_event_count = sum(int(row.get("output_fragment_event_count") or 0) for row in frame_diagnostics)
    output_fragment_merge_event_count = sum(
        int(row.get("output_fragment_merge_event_count") or 0) for row in frame_diagnostics
    )
    output_fragment_suppress_event_count = sum(
        int(row.get("output_fragment_suppress_event_count") or 0) for row in frame_diagnostics
    )
    prompt_with_relaxed_mask_count = sum(
        int((row.get("gap_stats") or {}).get("prompt_with_relaxed_mask_count") or 0)
        for row in frame_diagnostics
    )
    relaxed_selected_mask_count = sum(
        int((row.get("gap_stats") or {}).get("relaxed_selected_mask_count") or 0)
        for row in frame_diagnostics
    )
    gap_small_quality_drop_count = sum(
        int(((row.get("gap_stats") or {}).get("small_mask_quality_filter") or {}).get("dropped_mask_count") or 0)
        for row in frame_diagnostics
    )
    gap_output_quality_drop_count = sum(
        int(((row.get("gap_stats") or {}).get("output_quality_filter") or {}).get("dropped_mask_count") or 0)
        for row in frame_diagnostics
    )
    gap_admission_input_count = sum(
        int((row.get("gap_admission_stats") or {}).get("input_mask_count") or 0)
        for row in frame_diagnostics
    )
    gap_admission_admitted_count = sum(
        int((row.get("gap_admission_stats") or {}).get("admitted_mask_count") or 0)
        for row in frame_diagnostics
    )
    gap_admission_output_only_count = sum(
        int((row.get("gap_admission_stats") or {}).get("output_only_mask_count") or 0)
        for row in frame_diagnostics
    )
    gap_anti_merge_rejected_count = sum(
        int((row.get("gap_admission_stats") or {}).get("anti_merge_rejected_count") or 0)
        for row in frame_diagnostics
    )
    summary = {
        "schema_version": "stream4d_v105_baseline_x_sam2_twostage_tracking_summary_v1",
        "variant": variant,
        "baseline_id": str(args.baseline_id),
        "config_path": str(args.config_path),
        "scene_id": str(args.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "frame_count": int(len(frame_ids)),
        "model_provider": str(models.get("model_provider", args.model_provider)),
        "segmentor": str(models.get("segmentor_name", args.segmentor_name)),
        "tracker": str(models.get("tracker_name", args.tracker_name)),
        "frame0_policy": {
            "stage1": {
                "num_pts": int(args.stage1_num_pts),
                "point_mode": str(args.stage1_point_mode),
                "grid_side": int(args.stage1_grid_side),
                "choice_policy": str(args.stage1_choice_policy),
                "pred_iou_thresh": float(args.stage1_iou_threshold),
                "stability_score_thresh": float(args.stage1_stability_threshold),
                "apply_box_nms": bool(args.stage1_apply_box_nms),
                "nms_score_type": str(args.stage1_nms_score_type),
            },
            "stage2_uncovered": {
                "num_pts": int(args.stage2_num_pts),
                "choice_policy": str(args.stage2_choice_policy),
                "pred_iou_thresh": float(args.stage2_iou_threshold),
                "stability_score_thresh": float(args.stage2_stability_threshold),
                "apply_box_nms": bool(args.stage2_apply_box_nms),
            },
        },
        "gap_policy": {
            "num_pts_active": int(args.gap_num_pts_active),
            "choice_policy": str(args.gap_choice_policy),
            "pred_iou_thresh": float(args.gap_iou_threshold),
            "stability_score_thresh": float(args.gap_stability_threshold),
            "apply_box_nms": bool(args.gap_apply_box_nms),
            "sampler": str(args.gap_sampler),
            "max_points": int(args.gap_max_points),
            "min_component_area": int(args.gap_min_component_area),
            "base_points_per_component": int(args.gap_base_points_per_component),
            "area_per_extra_point": int(args.gap_area_per_extra_point),
            "max_points_per_component": int(args.gap_max_points_per_component),
            "min_image_edge_distance_px": int(args.gap_min_image_edge_distance_px),
            "relaxed_min_uncovered_ratio": float(
                getattr(args, "gap_relaxed_min_uncovered_ratio", 0.0)
            ),
            "relaxed_iou_threshold": float(getattr(args, "gap_relaxed_iou_threshold", 0.0)),
            "relaxed_stability_threshold": float(
                getattr(args, "gap_relaxed_stability_threshold", 0.0)
            ),
            "relaxed_min_clipped_area": int(getattr(args, "gap_relaxed_min_clipped_area", 0)),
            "small_mask_max_area": int(getattr(args, "gap_small_mask_max_area", 0)),
            "small_mask_min_pred_iou": float(getattr(args, "gap_small_mask_min_pred_iou", 0.0)),
            "output_min_pred_iou": float(getattr(args, "gap_output_min_pred_iou", 0.0)),
            "output_min_stability": float(getattr(args, "gap_output_min_stability", 0.0)),
            "output_allow_relaxed": bool(getattr(args, "gap_output_allow_relaxed", True)),
            "reuse_recent_id_window": int(getattr(args, "gap_reuse_recent_id_window", 0)),
            "reuse_recent_id_iou": float(getattr(args, "gap_reuse_recent_id_iou", 0.0)),
            "reuse_recent_id_min_area": int(getattr(args, "gap_reuse_recent_id_min_area", 0)),
            "large_reuse_recent_id_iou": float(getattr(args, "gap_large_reuse_recent_id_iou", 0.0)),
            "large_reuse_min_area": int(getattr(args, "gap_large_reuse_min_area", 0)),
            "large_reuse_max_area_ratio": float(getattr(args, "gap_large_reuse_max_area_ratio", 0.0)),
            "anti_merge_core_window_frames": int(getattr(args, "gap_anti_merge_core_window_frames", 0)),
            "anti_merge_core_erode_px": int(getattr(args, "gap_anti_merge_core_erode_px", 0)),
            "anti_merge_core_min_area": int(getattr(args, "gap_anti_merge_core_min_area", 0)),
            "anti_merge_core_min_overlap_px": int(getattr(args, "gap_anti_merge_core_min_overlap_px", 0)),
            "anti_merge_core_min_overlap_ratio": float(
                getattr(args, "gap_anti_merge_core_min_overlap_ratio", 0.0)
            ),
            "anti_merge_max_overlap_objects": int(getattr(args, "gap_anti_merge_max_overlap_objects", 1)),
        },
        "propagation_mode": str(args.propagation_mode),
        "propagation_chunk_size": int(args.propagation_chunk_size),
        "point_sampling_stage1": s1_point_meta,
        "points_per_batch": int(args.points_per_batch),
        "box_nms_thresh": float(args.box_nms_thresh),
        "empty_ratio": float(args.empty_ratio),
        "setup_sec": float(setup_sec),
        "total_runtime_sec": float(total_sec),
        "frame0_stage1_runtime_sec": float(stage1_sec),
        "frame0_stage2_runtime_sec": float(stage2_sec),
        "total_tracking_runtime_sec": float(total_tracking_sec),
        "total_gap_segmentation_runtime_sec": float(total_gap_seg_sec),
        "empty_propagation_frames": int(empty_propagation_frames),
        "gap_reuse_event_count": int(gap_reuse_event_count),
        "output_relabel_event_count": int(output_relabel_event_count),
        "output_relabel_collision_guard_event_count": int(output_relabel_collision_guard_event_count),
        "output_fragment_event_count": int(output_fragment_event_count),
        "output_fragment_merge_event_count": int(output_fragment_merge_event_count),
        "output_fragment_suppress_event_count": int(output_fragment_suppress_event_count),
        "prompt_with_relaxed_mask_count": int(prompt_with_relaxed_mask_count),
        "relaxed_selected_mask_count": int(relaxed_selected_mask_count),
        "gap_small_quality_drop_count": int(gap_small_quality_drop_count),
        "gap_output_quality_drop_count": int(gap_output_quality_drop_count),
        "gap_admission_input_count": int(gap_admission_input_count),
        "gap_admission_admitted_count": int(gap_admission_admitted_count),
        "gap_admission_output_only_count": int(gap_admission_output_only_count),
        "gap_anti_merge_rejected_count": int(gap_anti_merge_rejected_count),
        "runtime_tuning": models.get("runtime_tuning", {}),
        "stream_state_memory_policy": {
            "stream_disjoin_policy": str(stream_disjoin_policy),
            "stream_disjoin_claim_dropped": bool(stream_disjoin_claim_dropped),
            "stream_disjoin_min_area_px": int(stream_disjoin_min_area_px),
            "stream_disjoin_recent_min_iou": float(stream_disjoin_recent_min_iou),
            "stream_disjoin_recent_max_area_growth": float(stream_disjoin_recent_max_area_growth),
            "keep_noncond_frames": int(getattr(args, "stream_keep_noncond_frames", 0)),
            "prune_invisible_after_frames": int(getattr(args, "stream_prune_invisible_after_frames", 0)),
            "prune_min_visible_area": int(getattr(args, "stream_prune_min_visible_area", 0)),
            "prune_protect_min_ever_area": int(stream_prune_protect_min_ever_area),
            "prune_protect_max_objects": int(stream_prune_protect_max_objects),
            "prune_max_visible_area": int(stream_max_visible_area_arg),
            "prune_max_visible_area_ratio": float(stream_max_visible_area_ratio),
            "effective_prune_max_visible_area": int(stream_max_visible_area),
            "oversized_prune_action": str(stream_oversized_prune_action),
            "empty_cache_every": int(getattr(args, "stream_empty_cache_every", 0)),
            "empty_cache_on_prune": bool(getattr(args, "stream_empty_cache_on_prune", True)),
            "runtime_max_cond_frames_in_attn": int(getattr(args, "runtime_max_cond_frames_in_attn", 0)),
        },
        "output_identity_policy": {
            "reuse_recent_id_window": int(getattr(args, "output_reuse_recent_id_window", 0)),
            "reuse_recent_id_iou": float(getattr(args, "output_reuse_recent_id_iou", 0.0)),
            "reuse_recent_id_min_area": int(getattr(args, "output_reuse_recent_id_min_area", 0)),
            "large_reuse_recent_id_iou": float(getattr(args, "output_large_reuse_recent_id_iou", 0.0)),
            "large_reuse_min_area": int(getattr(args, "output_large_reuse_min_area", 0)),
            "large_reuse_max_area_ratio": float(getattr(args, "output_large_reuse_max_area_ratio", 0.0)),
            "canonical_preference": str(getattr(args, "output_reuse_recent_id_preference", "lower_id")),
            "prevent_collision_union": bool(getattr(args, "output_reuse_prevent_collision_union", False)),
        },
        "output_fragment_policy": {
            "fragment_max_area": int(getattr(args, "output_fragment_max_area", 0)),
            "suppress_max_area": int(getattr(args, "output_fragment_suppress_max_area", 0)),
            "merge_dilate_px": int(getattr(args, "output_fragment_merge_dilate_px", 2)),
            "merge_min_touch_px": int(getattr(args, "output_fragment_merge_min_touch_px", 16)),
            "merge_min_touch_ratio": float(getattr(args, "output_fragment_merge_min_touch_ratio", 0.02)),
            "merge_min_neighbor_area": int(getattr(args, "output_fragment_merge_min_neighbor_area", 20000)),
        },
        "gap_admission_policy": {
            "delayed_admission_enabled": bool(getattr(args, "gap_delayed_admission_enabled", False)),
            "min_area": int(getattr(args, "birth_admission_min_area", 0)),
            "max_area": int(getattr(args, "birth_admission_max_area", 0)),
            "max_uncovered_ratio": float(getattr(args, "birth_admission_max_uncovered_ratio", 0.0)),
            "max_bbox_frac": float(getattr(args, "birth_admission_max_bbox_frac", 0.0)),
            "max_edge_touch_count": int(getattr(args, "birth_admission_max_edge_touch_count", -1)),
            "min_extent": float(getattr(args, "birth_admission_min_extent", 0.0)),
            "min_core_area": int(getattr(args, "birth_admission_min_core_area_px", 0)),
            "shape_min_uncovered_ratio": float(getattr(args, "birth_admission_shape_min_uncovered_ratio", 0.0)),
            "max_per_frame": int(getattr(args, "birth_admission_max_per_frame", 0)),
            "min_pred_iou": float(getattr(args, "gap_admission_min_pred_iou", 0.0)),
            "min_stability": float(getattr(args, "gap_admission_min_stability", 0.0)),
            "allow_relaxed": bool(getattr(args, "gap_admission_allow_relaxed", True)),
        },
        "stream_memory_prune_events": stream_memory_prune_events,
        "stream_object_prune_events": stream_object_prune_events,
        "stream_protected_invisible_events": stream_protected_invisible_events,
        "final_active_stream_object_count": int(len(stream_state.get("obj_ids", []))) if stream_state is not None else 0,
        "final_noncond_stream_frame_count": int(
            len(stream_state.get("output_dict", {}).get("non_cond_frame_outputs", {}))
        )
        if stream_state is not None
        else 0,
        "initial_stage1_mask_count": int(stage1_masks.shape[0]),
        "initial_stage2_mask_count": int(stage2_masks.shape[0]),
        "initial_mask_count": int(masks0.shape[0]),
        "total_object_id_count": int(next_obj_id),
        "mean_visible_id_count": float(np.mean([row["visible_id_count"] for row in records])) if records else 0.0,
        "mean_foreground_ratio": float(np.mean([row["foreground_ratio"] for row in records])) if records else 0.0,
        "peak_cuda_memory_mb": float(peak_mb),
        "segmentor_checkpoint": str(models["segmentor_checkpoint"]),
        "segmentor_checkpoint_sha256": sha256_file(Path(models["segmentor_checkpoint"])),
        "segmentor_cfg": str(models["segmentor_cfg"]),
        "sam2_checkpoint": str(models["sam2_checkpoint"]),
        "sam2_checkpoint_sha256": sha256_file(Path(models["sam2_checkpoint"])),
        "sam2_cfg": str(models["sam2_cfg"]),
        "tracker_checkpoint": str(models["tracker_checkpoint"]),
        "tracker_checkpoint_sha256": sha256_file(Path(models["tracker_checkpoint"])),
        "tracker_cfg": str(models["tracker_cfg"]),
        "video_path": str(video_path),
        "sheet_paths": sheet_paths,
        "records": records,
        "frame_diagnostics": frame_diagnostics,
    }
    if birth_dump_dir is not None:
        birth_records_path = birth_dump_dir / "birth_records.json"
        birth_payload = {
            "schema_version": "stream4d_v105_birth_mask_dump_v1",
            "scene_id": str(args.scene_id),
            "frame_ids": [int(v) for v in frame_ids],
            "row_count": len(birth_records),
            "rows": birth_records,
        }
        birth_records_path.write_text(json.dumps(birth_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["birth_dump_dir"] = str(birth_dump_dir)
        summary["birth_records_json"] = str(birth_records_path)
        summary["birth_record_count"] = int(len(birth_records))
    if "efficient_sam2_bypass_checkpoint" in models:
        summary["efficient_sam2_bypass_checkpoint"] = str(models["efficient_sam2_bypass_checkpoint"])
        summary["efficient_sam2_bypass_checkpoint_sha256"] = sha256_file(Path(models["efficient_sam2_bypass_checkpoint"]))
    if "efficient_sam2_root" in models:
        summary["efficient_sam2_root"] = str(models["efficient_sam2_root"])
    if "efficient_sam2_runtime" in models:
        summary["efficient_sam2_runtime"] = models["efficient_sam2_runtime"]
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "video": str(video_path), "sheets": sheet_paths}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--rgb-root", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--birth-dump-dir", default="")
    return parser


def main() -> None:
    cli = build_parser().parse_args()
    config_path = Path(cli.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cli.config = str(config_path)
    config = load_config(config_path)
    args = make_args(config, cli)
    run(args)


if __name__ == "__main__":
    main()
