#!/usr/bin/env python3
"""v106 hybrid adapter: SAM2 image segmentor plus EdgeTAM video tracker."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base
import tools.audit_v106_edgetam_twostage_tracking as edge


REPO_ROOT = base.REPO_ROOT
load_config = base.load_config
get_runtime_stats = edge.get_runtime_stats


def make_args(config: dict[str, Any], cli: Any) -> SimpleNamespace:
    args = edge.make_args(config, cli)
    paths = config.get("paths", {})
    args.baseline_id = str(base.cfg_get(config, "baseline", "id", default="v106-stateful-sam2seg-edgetam-tracker"))
    args.variant_id = str(base.cfg_get(config, "baseline", "variant", default="v106_stateful_sam2seg_edgetam_tracker"))
    args.model_provider = "sam2seg_edgetam_tracker"
    args.segmentor_name = str(base.cfg_get(config, "baseline", "segmentor", default="sam2.1_hiera_large"))
    args.tracker_name = "edgetam"
    args.tracker_backend = "edgetam"
    args.tracker = "edgetam"
    args.sam2_checkpoint = str(paths.get("sam2_checkpoint", "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"))
    args.sam2_model_cfg = str(paths.get("sam2_model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"))
    args.edgetam_root = str(paths.get("edgetam_root", "third_party/EdgeTAM"))
    args.edgetam_checkpoint = str(paths.get("edgetam_checkpoint", "third_party/EdgeTAM/checkpoints/edgetam.pt"))
    args.edgetam_model_cfg = str(paths.get("edgetam_model_cfg", "edgetam.yaml"))
    return args


def _load_sam2_segmentor(args: SimpleNamespace) -> dict[str, Any]:
    repo_path = REPO_ROOT / "Grounded-SAM-2"
    base.clear_sam2_namespace(repo_path)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = (REPO_ROOT / args.sam2_checkpoint).resolve() if not Path(args.sam2_checkpoint).is_absolute() else Path(args.sam2_checkpoint)
    cfg = str(args.sam2_model_cfg)
    image_model = build_sam2(cfg, str(checkpoint), device="cuda")
    segmentor = SAM2ImagePredictor(image_model)
    tuning = base.apply_runtime_model_tuning(image_model, args)
    return {
        "segmentor": segmentor,
        "segmentor_checkpoint": checkpoint,
        "segmentor_cfg": cfg,
        "segmentor_tuning": tuning,
    }


def _load_edgetam_tracker(args: SimpleNamespace) -> dict[str, Any]:
    from hydra.core.global_hydra import GlobalHydra

    root = (REPO_ROOT / args.edgetam_root).resolve() if not Path(args.edgetam_root).is_absolute() else Path(args.edgetam_root)
    checkpoint = (
        (REPO_ROOT / args.edgetam_checkpoint).resolve()
        if not Path(args.edgetam_checkpoint).is_absolute()
        else Path(args.edgetam_checkpoint)
    )
    if not root.exists():
        raise FileNotFoundError(f"missing EdgeTAM root: {root}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing EdgeTAM checkpoint: {checkpoint}")

    cfg = str(args.edgetam_model_cfg)
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    base.clear_sam2_namespace(root)
    try:
        os.chdir(root)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        from sam2.build_sam import build_sam2_video_predictor

        tracker_model = build_sam2_video_predictor(cfg, str(checkpoint), device="cuda")
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
    tuning = base.apply_runtime_model_tuning(tracker_model, args)
    return {
        "tracker_model": tracker_model,
        "tracker_checkpoint": checkpoint,
        "tracker_cfg": cfg,
        "tracker_tuning": tuning,
        "edgetam_root": root,
    }


def setup_models(args: SimpleNamespace) -> dict[str, Any]:
    sam2 = _load_sam2_segmentor(args)
    edgetam = _load_edgetam_tracker(args)
    return {
        "model_provider": "sam2seg_edgetam_tracker",
        "segmentor": sam2["segmentor"],
        "tracker_model": edgetam["tracker_model"],
        "segmentor_name": str(args.segmentor_name),
        "tracker_name": "edgetam",
        "segmentor_checkpoint": sam2["segmentor_checkpoint"],
        "segmentor_cfg": sam2["segmentor_cfg"],
        "tracker_checkpoint": edgetam["tracker_checkpoint"],
        "tracker_cfg": edgetam["tracker_cfg"],
        "sam2_checkpoint": sam2["segmentor_checkpoint"],
        "sam2_cfg": sam2["segmentor_cfg"],
        "edgetam_root": edgetam["edgetam_root"],
        "edgetam_checkpoint": edgetam["tracker_checkpoint"],
        "edgetam_cfg": edgetam["tracker_cfg"],
        "runtime_tuning": {
            "segmentor": sam2["segmentor_tuning"],
            "tracker": edgetam["tracker_tuning"],
        },
    }


def run(args: SimpleNamespace) -> None:
    import torch

    edge.reset_runtime_stats()
    setattr(edge.add_masks_to_stream_state, "_args", args)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.empty((1,), device="cuda")
    original_setup_models = base.setup_models
    original_add_masks = base.add_masks_to_stream_state
    try:
        base.setup_models = setup_models
        base.add_masks_to_stream_state = edge.add_masks_to_stream_state
        base.run(args)
    finally:
        if hasattr(edge.add_masks_to_stream_state, "_args"):
            delattr(edge.add_masks_to_stream_state, "_args")
        base.setup_models = original_setup_models
        base.add_masks_to_stream_state = original_add_masks
