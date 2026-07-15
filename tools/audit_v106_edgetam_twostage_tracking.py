#!/usr/bin/env python3
"""EdgeTAM-only adapter for v106 two-stage segmentation and stateful tracking.

This module keeps EdgeTAM provider wiring out of the v105 baseline-x SAM2
runner. It reuses the baseline loop, point sampling, mask disjoining, and
visual export, but owns the EdgeTAM model setup and EdgeTAM add-mask behavior.
"""

from __future__ import annotations

import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base


REPO_ROOT = base.REPO_ROOT
load_config = base.load_config

_RUNTIME_STATS: dict[str, Any] = {}


def reset_runtime_stats() -> None:
    global _RUNTIME_STATS
    _RUNTIME_STATS = {
        "schema_version": "stream4d_v106_edgetam_runtime_stats_v1",
        "memory_admission_calls": 0,
        "initial_memory_admission_calls": 0,
        "gap_memory_admission_calls": 0,
        "requested_mask_count": 0,
        "admitted_mask_count": 0,
        "skipped_mask_count": 0,
        "add_new_mask_runtime_sec": 0.0,
        "reconsolidate_runtime_sec": 0.0,
        "memory_admission_runtime_sec": 0.0,
        "reconsolidate_call_count": 0,
        "records": [],
    }


def get_runtime_stats() -> dict[str, Any]:
    if not _RUNTIME_STATS:
        reset_runtime_stats()
    return deepcopy(_RUNTIME_STATS)


def make_args(config: dict[str, Any], cli: Any) -> SimpleNamespace:
    args = base.make_args(config, cli)
    paths = config.get("paths", {})
    args.baseline_id = str(base.cfg_get(config, "baseline", "id", default="v106-stateful-edgetam-scene-stream"))
    args.variant_id = str(base.cfg_get(config, "baseline", "variant", default="v106_stateful_edgetam_scene_stream"))
    args.model_provider = "edgetam"
    args.segmentor_name = "edgetam"
    args.tracker_name = "edgetam"
    args.tracker_backend = "edgetam"
    args.tracker = "edgetam"
    args.edgetam_root = str(paths.get("edgetam_root", "third_party/EdgeTAM"))
    args.edgetam_checkpoint = str(paths.get("edgetam_checkpoint", "third_party/EdgeTAM/checkpoints/edgetam.pt"))
    args.edgetam_model_cfg = str(paths.get("edgetam_model_cfg", "edgetam.yaml"))
    edge_gap = config.get("edgetam_gap", {})
    yoloe_gap = config.get("yoloe_gap", {})
    args.edgetam_gap_birth_min_area = int(edge_gap.get("birth_min_area", 0))
    args.edgetam_gap_birth_max_area_ratio = float(edge_gap.get("birth_max_area_ratio", 0.0))
    args.edgetam_gap_birth_max_masks_per_frame = int(edge_gap.get("birth_max_masks_per_frame", 0))
    args.edgetam_gap_birth_sort_by = str(edge_gap.get("birth_sort_by", "input_order"))
    args.edgetam_gap_birth_interval = int(edge_gap.get("birth_interval", 1))
    args.edgetam_gap_birth_warmup_frames = int(edge_gap.get("birth_warmup_frames", 0))
    args.edgetam_gap_birth_force_guide_area_ratio = float(edge_gap.get("birth_force_guide_area_ratio", 0.0))
    memory_cfg = config.get("edgetam_memory", {})
    args.edgetam_memory_admission_policy = str(memory_cfg.get("admission_policy", "all"))
    args.edgetam_memory_admit_min_area = int(memory_cfg.get("admit_min_area", 0))
    args.edgetam_memory_admit_interval = int(memory_cfg.get("admit_interval", 1))
    args.edgetam_memory_admit_warmup_frames = int(memory_cfg.get("admit_warmup_frames", 0))
    args.edgetam_memory_admit_force_area_ratio = float(memory_cfg.get("admit_force_area_ratio", 0.0))
    args.edgetam_memory_admit_max_masks_per_frame = int(memory_cfg.get("admit_max_masks_per_frame", 0))
    args.edgetam_memory_admit_sort_by = str(memory_cfg.get("admit_sort_by", "area_desc"))
    args.yoloe_gap_enabled = bool(yoloe_gap.get("enabled", False))
    args.yoloe_gap_model = str(yoloe_gap.get("model", "yoloe-11l-seg.pt"))
    args.yoloe_gap_prompt_free = bool(yoloe_gap.get("prompt_free", "-pf" in args.yoloe_gap_model))
    prompts_raw = yoloe_gap.get("prompts", [])
    if isinstance(prompts_raw, str):
        prompts = [item.strip() for item in prompts_raw.split(",") if item.strip()]
    elif isinstance(prompts_raw, list):
        prompts = [str(item).strip() for item in prompts_raw if str(item).strip()]
    else:
        prompts = []
    args.yoloe_gap_prompts = prompts
    args.yoloe_gap_device = str(yoloe_gap.get("device", "cuda"))
    args.yoloe_gap_conf = float(yoloe_gap.get("conf", 0.2))
    args.yoloe_gap_iou = float(yoloe_gap.get("iou", 0.5))
    args.yoloe_gap_imgsz = int(yoloe_gap.get("imgsz", 0))
    args.yoloe_gap_max_detections = int(yoloe_gap.get("max_detections", 24))
    args.yoloe_gap_min_guide_area = int(yoloe_gap.get("min_guide_area", 600))
    args.yoloe_gap_max_guide_area_ratio = float(yoloe_gap.get("max_guide_area_ratio", 0.0))
    args.yoloe_gap_points_per_detection = int(yoloe_gap.get("points_per_detection", 1))
    args.yoloe_gap_use_masks = bool(yoloe_gap.get("use_masks", True))
    args.yoloe_gap_use_boxes = bool(yoloe_gap.get("use_boxes", True))
    args.yoloe_gap_box_expand_ratio = float(yoloe_gap.get("box_expand_ratio", 0.06))
    args.yoloe_gap_guide_point_fraction = float(yoloe_gap.get("guide_point_fraction", 0.65))
    args.yoloe_gap_fallback_to_uncovered = bool(yoloe_gap.get("fallback_to_uncovered", True))
    args.yoloe_gap_fallback_min_component_area = int(yoloe_gap.get("fallback_min_component_area", 2500))
    args.yoloe_gap_exclude_guided_from_fallback = bool(yoloe_gap.get("exclude_guided_from_fallback", True))
    return args


def setup_models(args: SimpleNamespace) -> dict[str, Any]:
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
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        image_model = build_sam2(cfg, str(checkpoint), device="cuda")
        segmentor = SAM2ImagePredictor(image_model)
        tracker_model = build_sam2_video_predictor(cfg, str(checkpoint), device="cuda")
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path

    segmentor_tuning = base.apply_runtime_model_tuning(image_model, args)
    tracker_tuning = base.apply_runtime_model_tuning(tracker_model, args)
    return {
        "model_provider": "edgetam",
        "segmentor": segmentor,
        "tracker_model": tracker_model,
        "segmentor_name": "edgetam",
        "tracker_name": "edgetam",
        "segmentor_checkpoint": checkpoint,
        "segmentor_cfg": cfg,
        "tracker_checkpoint": checkpoint,
        "tracker_cfg": cfg,
        "sam2_checkpoint": checkpoint,
        "sam2_cfg": cfg,
        "edgetam_root": root,
        "edgetam_checkpoint": checkpoint,
        "edgetam_cfg": cfg,
        "runtime_tuning": {
            "segmentor": segmentor_tuning,
            "tracker": tracker_tuning,
        },
    }


def _select_memory_admission_masks(
    args: SimpleNamespace,
    *,
    frame_idx: int,
    obj_ids: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if masks.size == 0:
        return obj_ids, masks, {
            "policy": str(getattr(args, "edgetam_memory_admission_policy", "all")),
            "frame_idx": int(frame_idx),
            "requested_mask_count": 0,
            "admitted_mask_count": 0,
            "skipped_mask_count": 0,
        }
    h, w = masks.shape[-2:]
    areas = masks.reshape((masks.shape[0], -1)).sum(axis=1).astype(np.int64)
    policy = str(getattr(args, "edgetam_memory_admission_policy", "all"))
    keep_indices = np.arange(masks.shape[0], dtype=np.int64)
    scheduled = True
    force_large = np.zeros((masks.shape[0],), dtype=bool)
    area_ok = np.ones((masks.shape[0],), dtype=bool)
    min_area = max(0, int(getattr(args, "edgetam_memory_admit_min_area", 0)))
    interval = max(1, int(getattr(args, "edgetam_memory_admit_interval", 1)))
    warmup_frames = max(0, int(getattr(args, "edgetam_memory_admit_warmup_frames", 0)))
    force_area_ratio = max(0.0, float(getattr(args, "edgetam_memory_admit_force_area_ratio", 0.0)))
    max_masks = max(0, int(getattr(args, "edgetam_memory_admit_max_masks_per_frame", 0)))
    sort_by = str(getattr(args, "edgetam_memory_admit_sort_by", "area_desc"))
    capped_count = 0

    if int(frame_idx) > 0 and policy not in {"", "all"}:
        scheduled = interval <= 1 or int(frame_idx) % interval == 0 or int(frame_idx) <= warmup_frames
        area_ok = areas >= int(min_area)
        ratios = areas.astype(np.float64) / float(max(h * w, 1))
        if force_area_ratio > 0.0:
            force_large = ratios >= float(force_area_ratio)
        if policy in {"area_scheduled", "scheduled_area", "scheduled_large"}:
            keep_mask = area_ok & (bool(scheduled) | force_large)
        elif policy == "large_only":
            keep_mask = area_ok & force_large
        elif policy == "none":
            keep_mask = np.zeros_like(area_ok, dtype=bool)
        else:
            keep_mask = np.ones_like(area_ok, dtype=bool)
        keep_indices = np.flatnonzero(keep_mask).astype(np.int64)
        if keep_indices.size:
            if sort_by == "area_desc":
                keep_indices = keep_indices[np.argsort(-areas[keep_indices])]
            elif sort_by == "area_asc":
                keep_indices = keep_indices[np.argsort(areas[keep_indices])]
        before_cap = int(keep_indices.size)
        if max_masks > 0:
            keep_indices = keep_indices[:max_masks]
            capped_count = max(0, before_cap - int(keep_indices.size))

    admitted_masks = masks[keep_indices].astype(bool) if keep_indices.size else np.zeros((0, h, w), dtype=bool)
    admitted_ids = obj_ids[keep_indices].astype(np.int64) if keep_indices.size else np.zeros((0,), dtype=np.int64)
    skipped = int(masks.shape[0]) - int(admitted_masks.shape[0])
    return admitted_ids, admitted_masks, {
        "policy": policy,
        "frame_idx": int(frame_idx),
        "requested_mask_count": int(masks.shape[0]),
        "admitted_mask_count": int(admitted_masks.shape[0]),
        "skipped_mask_count": int(skipped),
        "scheduled": bool(scheduled),
        "admit_interval": int(interval),
        "admit_warmup_frames": int(warmup_frames),
        "admit_min_area": int(min_area),
        "admit_force_area_ratio": float(force_area_ratio),
        "admit_max_masks_per_frame": int(max_masks),
        "admit_sort_by": sort_by,
        "force_large_count": int(np.count_nonzero(force_large & area_ok)),
        "rejected_small_count": int(np.count_nonzero(~area_ok)) if int(frame_idx) > 0 else 0,
        "skipped_unscheduled_count": int(np.count_nonzero(area_ok & ~force_large)) if int(frame_idx) > 0 and not bool(scheduled) else 0,
        "capped_count": int(capped_count),
        "requested_areas": [int(v) for v in areas[:16]],
        "admitted_areas": [int(areas[int(i)]) for i in keep_indices[:16]],
    }


def _record_memory_admission(record: dict[str, Any]) -> None:
    if not _RUNTIME_STATS:
        reset_runtime_stats()
    stats = _RUNTIME_STATS
    stats["memory_admission_calls"] = int(stats["memory_admission_calls"]) + 1
    if int(record.get("frame_idx", 0)) <= 0:
        stats["initial_memory_admission_calls"] = int(stats["initial_memory_admission_calls"]) + 1
    else:
        stats["gap_memory_admission_calls"] = int(stats["gap_memory_admission_calls"]) + 1
    stats["requested_mask_count"] = int(stats["requested_mask_count"]) + int(record.get("requested_mask_count", 0))
    stats["admitted_mask_count"] = int(stats["admitted_mask_count"]) + int(record.get("admitted_mask_count", 0))
    stats["skipped_mask_count"] = int(stats["skipped_mask_count"]) + int(record.get("skipped_mask_count", 0))
    stats["add_new_mask_runtime_sec"] = float(stats["add_new_mask_runtime_sec"]) + float(record.get("add_new_mask_runtime_sec", 0.0))
    stats["reconsolidate_runtime_sec"] = float(stats["reconsolidate_runtime_sec"]) + float(record.get("reconsolidate_runtime_sec", 0.0))
    stats["memory_admission_runtime_sec"] = float(stats["memory_admission_runtime_sec"]) + float(record.get("memory_admission_runtime_sec", 0.0))
    stats["reconsolidate_call_count"] = int(stats["reconsolidate_call_count"]) + int(record.get("reconsolidate_called", False))
    records = stats.setdefault("records", [])
    if len(records) < 512:
        records.append(record)


def add_masks_to_stream_state(
    predictor: Any,
    state: dict[str, Any],
    *,
    tracker: str,
    frame_idx: int,
    obj_ids: np.ndarray,
    masks: np.ndarray,
) -> None:
    """Add masks to an EdgeTAM state with bf16 autocast for post-start births."""
    import torch
    from contextlib import nullcontext

    started = time.time()
    if masks.size == 0:
        args = getattr(add_masks_to_stream_state, "_args", SimpleNamespace())
        _record_memory_admission(
            {
                "frame_idx": int(frame_idx),
                "policy": str(getattr(args, "edgetam_memory_admission_policy", "all")),
                "requested_mask_count": 0,
                "admitted_mask_count": 0,
                "skipped_mask_count": 0,
                "add_new_mask_runtime_sec": 0.0,
                "reconsolidate_runtime_sec": 0.0,
                "memory_admission_runtime_sec": float(time.time() - started),
                "reconsolidate_called": False,
            }
        )
        return
    obj_ids, masks, admission_meta = _select_memory_admission_masks(
        getattr(add_masks_to_stream_state, "_args", SimpleNamespace()),
        frame_idx=int(frame_idx),
        obj_ids=obj_ids,
        masks=masks,
    )
    if masks.size == 0:
        admission_meta.update(
            {
                "add_new_mask_runtime_sec": 0.0,
                "reconsolidate_runtime_sec": 0.0,
                "memory_admission_runtime_sec": float(time.time() - started),
                "reconsolidate_called": False,
            }
        )
        _record_memory_admission(admission_meta)
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
        state.get("frames_already_tracked", {}).pop(int(frame_idx), None)
    old_tracking_started = bool(state.get("tracking_has_started", False))
    state["tracking_has_started"] = False
    add_sec = 0.0
    try:
        for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
            mask_arg: Any = torch.from_numpy(mask.astype(np.float32))
            autocast_ctx = (
                torch.autocast("cuda", dtype=autocast_dtype)
                if autocast_dtype in {torch.bfloat16, torch.float16}
                else nullcontext()
            )
            with torch.inference_mode(), autocast_ctx:
                add_t0 = time.time()
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=int(obj_id),
                    mask=mask_arg,
                )
                add_sec += float(time.time() - add_t0)
    finally:
        state["tracking_has_started"] = old_tracking_started
    new_obj_count = len(state.get("obj_ids", []))
    reconsolidate_sec = 0.0
    reconsolidate_called = False
    if old_tracking_started and new_obj_count > old_obj_count:
        reconsolidate_called = True
        recon_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            base.reconsolidate_stream_state_outputs(predictor, state)
        reconsolidate_sec = float(time.time() - recon_t0)
    admission_meta.update(
        {
            "old_tracking_started": bool(old_tracking_started),
            "old_object_count": int(old_obj_count),
            "new_object_count": int(new_obj_count),
            "add_new_mask_runtime_sec": float(add_sec),
            "reconsolidate_runtime_sec": float(reconsolidate_sec),
            "memory_admission_runtime_sec": float(time.time() - started),
            "reconsolidate_called": bool(reconsolidate_called),
        }
    )
    _record_memory_admission(admission_meta)


def _filter_gap_birth_masks(masks: np.ndarray, args: SimpleNamespace) -> tuple[np.ndarray, dict[str, Any]]:
    if masks.size == 0:
        return masks, {
            "enabled": True,
            "input_mask_count": 0,
            "output_mask_count": 0,
        }
    h, w = masks.shape[-2:]
    areas = masks.reshape((masks.shape[0], -1)).sum(axis=1).astype(np.int64)
    keep = np.ones((masks.shape[0],), dtype=bool)
    min_area = max(0, int(getattr(args, "edgetam_gap_birth_min_area", 0)))
    max_area_ratio = float(getattr(args, "edgetam_gap_birth_max_area_ratio", 0.0))
    if min_area > 0:
        keep &= areas >= int(min_area)
    if max_area_ratio > 0.0:
        keep &= (areas.astype(np.float64) / float(max(h * w, 1))) <= float(max_area_ratio)
    keep_indices = np.flatnonzero(keep)
    sort_by = str(getattr(args, "edgetam_gap_birth_sort_by", "input_order"))
    if sort_by == "area_desc" and keep_indices.size:
        keep_indices = keep_indices[np.argsort(-areas[keep_indices])]
    elif sort_by == "area_asc" and keep_indices.size:
        keep_indices = keep_indices[np.argsort(areas[keep_indices])]
    max_masks = int(getattr(args, "edgetam_gap_birth_max_masks_per_frame", 0))
    if max_masks > 0:
        keep_indices = keep_indices[:max_masks]
    filtered = masks[keep_indices].astype(bool) if keep_indices.size else np.zeros((0, h, w), dtype=bool)
    return filtered, {
        "enabled": True,
        "input_mask_count": int(masks.shape[0]),
        "output_mask_count": int(filtered.shape[0]),
        "rejected_small_count": int(np.count_nonzero(areas < int(min_area))) if min_area > 0 else 0,
        "rejected_large_count": int(np.count_nonzero((areas.astype(np.float64) / float(max(h * w, 1))) > float(max_area_ratio))) if max_area_ratio > 0.0 else 0,
        "min_area": int(min_area),
        "max_area_ratio": float(max_area_ratio),
        "max_masks_per_frame": int(max_masks),
        "sort_by": sort_by,
        "kept_areas": [int(areas[int(i)]) for i in keep_indices[:16]],
    }


class _YoloeGapGuide:
    def __init__(self, args: SimpleNamespace) -> None:
        self.args = args
        self.model: Any | None = None
        self.available = False
        self.load_error = ""

    def _resolve_model_path(self) -> Path:
        path = Path(str(getattr(self.args, "yoloe_gap_model", "")))
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path.resolve()

    def _ensure_model(self) -> None:
        if self.model is not None or self.load_error:
            return
        model_path = self._resolve_model_path()
        if not model_path.exists():
            self.load_error = f"missing YOLOE model: {model_path}"
            return
        try:
            from ultralytics import YOLOE

            self.model = YOLOE(str(model_path))
            prompts = list(getattr(self.args, "yoloe_gap_prompts", []) or [])
            prompt_free = bool(getattr(self.args, "yoloe_gap_prompt_free", False))
            if prompts and not prompt_free:
                self.model.set_classes(prompts, self.model.get_text_pe(prompts))
            device = str(getattr(self.args, "yoloe_gap_device", "cuda"))
            if device.startswith("cuda"):
                self.model.to("cuda")
            self.available = True
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.model = None
            self.available = False

    def build_guide_mask(self, rgb: np.ndarray, uncovered: np.ndarray) -> tuple[np.ndarray | None, list[np.ndarray], dict[str, Any]]:
        import cv2

        started = time.time()
        self._ensure_model()
        h, w = uncovered.shape
        meta: dict[str, Any] = {
            "enabled": True,
            "available": bool(self.available),
            "model": str(getattr(self.args, "yoloe_gap_model", "")),
            "prompt_free": bool(getattr(self.args, "yoloe_gap_prompt_free", False)),
            "prompt_count": int(len(getattr(self.args, "yoloe_gap_prompts", []) or [])),
            "load_error": self.load_error,
            "runtime_sec": 0.0,
            "raw_detection_count": 0,
            "used_mask_count": 0,
            "used_box_count": 0,
            "rejected_small_guide_count": 0,
            "rejected_large_guide_count": 0,
            "candidate_mask_count": 0,
            "candidate_area_max": 0,
            "candidate_areas": [],
            "guide_area": 0,
        }
        if self.model is None:
            meta["runtime_sec"] = float(time.time() - started)
            return None, [], meta
        try:
            device = str(getattr(self.args, "yoloe_gap_device", "cuda"))
            device_arg: Any = 0 if device.startswith("cuda") else device
            kwargs: dict[str, Any] = {
                "conf": float(getattr(self.args, "yoloe_gap_conf", 0.2)),
                "iou": float(getattr(self.args, "yoloe_gap_iou", 0.5)),
                "verbose": False,
                "device": device_arg,
            }
            imgsz = int(getattr(self.args, "yoloe_gap_imgsz", 0))
            if imgsz > 0:
                kwargs["imgsz"] = imgsz
            results = self.model.predict(rgb, **kwargs)
            result = results[0] if results else None
        except Exception as exc:
            meta["runtime_sec"] = float(time.time() - started)
            meta["predict_error"] = f"{type(exc).__name__}: {exc}"
            return None, [], meta
        if result is None:
            meta["runtime_sec"] = float(time.time() - started)
            return None, [], meta

        guide = np.zeros((h, w), dtype=bool)
        candidate_masks: list[np.ndarray] = []
        candidate_areas: list[int] = []
        min_area = max(0, int(getattr(self.args, "yoloe_gap_min_guide_area", 600)))
        max_area_ratio = max(0.0, float(getattr(self.args, "yoloe_gap_max_guide_area_ratio", 0.0)))
        max_area = int(round(float(h * w) * max_area_ratio)) if max_area_ratio > 0.0 else 0
        max_det = max(1, int(getattr(self.args, "yoloe_gap_max_detections", 24)))
        raw_count = 0
        boxes_xyxy: np.ndarray | None = None
        if getattr(result, "boxes", None) is not None and result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            raw_count = int(boxes_xyxy.shape[0])
        meta["raw_detection_count"] = int(raw_count)

        if bool(getattr(self.args, "yoloe_gap_use_masks", True)) and getattr(result, "masks", None) is not None and result.masks is not None:
            mask_data = result.masks.data.detach().cpu().numpy()
            for mask in mask_data[:max_det]:
                resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
                candidate = resized & uncovered.astype(bool)
                area = int(np.count_nonzero(candidate))
                if area < min_area:
                    meta["rejected_small_guide_count"] = int(meta["rejected_small_guide_count"]) + 1
                    continue
                if max_area > 0 and area > max_area:
                    meta["rejected_large_guide_count"] = int(meta["rejected_large_guide_count"]) + 1
                    continue
                guide |= candidate
                candidate_masks.append(candidate)
                candidate_areas.append(int(area))
                meta["used_mask_count"] = int(meta["used_mask_count"]) + 1

        if bool(getattr(self.args, "yoloe_gap_use_boxes", True)) and boxes_xyxy is not None and (not np.any(guide) or not bool(getattr(self.args, "yoloe_gap_use_masks", True))):
            expand = float(getattr(self.args, "yoloe_gap_box_expand_ratio", 0.06))
            for box in boxes_xyxy[:max_det]:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                x1i = max(0, int(round(x1 - bw * expand)))
                y1i = max(0, int(round(y1 - bh * expand)))
                x2i = min(w, int(round(x2 + bw * expand)))
                y2i = min(h, int(round(y2 + bh * expand)))
                if x2i <= x1i or y2i <= y1i:
                    continue
                candidate = np.zeros((h, w), dtype=bool)
                candidate[y1i:y2i, x1i:x2i] = True
                candidate &= uncovered.astype(bool)
                area = int(np.count_nonzero(candidate))
                if area < min_area:
                    meta["rejected_small_guide_count"] = int(meta["rejected_small_guide_count"]) + 1
                    continue
                if max_area > 0 and area > max_area:
                    meta["rejected_large_guide_count"] = int(meta["rejected_large_guide_count"]) + 1
                    continue
                guide |= candidate
                candidate_masks.append(candidate)
                candidate_areas.append(int(area))
                meta["used_box_count"] = int(meta["used_box_count"]) + 1

        meta["candidate_mask_count"] = int(len(candidate_masks))
        meta["candidate_area_max"] = int(max(candidate_areas) if candidate_areas else 0)
        meta["candidate_areas"] = [int(v) for v in candidate_areas[:32]]
        meta["guide_area"] = int(np.count_nonzero(guide))
        meta["runtime_sec"] = float(time.time() - started)
        return (guide if np.any(guide) else None), candidate_masks, meta


def _make_yoloe_guided_sampler(args: SimpleNamespace, original_sampler: Any) -> Any:
    if not bool(getattr(args, "yoloe_gap_enabled", False)):
        return original_sampler

    import torch

    frame_ids = base.parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = (REPO_ROOT / args.rgb_root).resolve() if not Path(args.rgb_root).is_absolute() else Path(args.rgb_root)
    rgb_root = rgb_root / args.scene_id / "color"
    guide = _YoloeGapGuide(args)
    call_state = {"gap_index": 0}

    def guided_sampler(
        mask_np: np.ndarray,
        *,
        max_points: int,
        min_component_area: int,
        base_points_per_component: int,
        area_per_extra_point: int,
        max_points_per_component: int,
        seed: int,
    ) -> tuple[Any, dict[str, Any]]:
        gap_index = int(call_state["gap_index"])
        call_state["gap_index"] = gap_index + 1
        frame_pos = min(gap_index + 1, max(len(frame_ids) - 1, 0))
        frame_id = int(frame_ids[frame_pos]) if frame_ids else int(frame_pos)
        rgb_path = rgb_root / f"{frame_id}.jpg"
        yoloe_meta: dict[str, Any] = {"enabled": True, "frame_id": int(frame_id), "rgb_path": str(rgb_path)}
        try:
            rgb = base.read_rgb(rgb_path)
            guide_mask, candidate_masks, guide_meta = guide.build_guide_mask(rgb, mask_np.astype(bool))
            yoloe_meta.update(guide_meta)
        except Exception as exc:
            guide_mask = None
            candidate_masks = []
            yoloe_meta.update({"available": False, "guide_error": f"{type(exc).__name__}: {exc}"})

        max_points_i = max(0, int(max_points))
        guide_fraction = min(1.0, max(0.0, float(getattr(args, "yoloe_gap_guide_point_fraction", 0.65))))
        guide_quota = int(round(float(max_points_i) * guide_fraction)) if guide_mask is not None else 0
        guide_quota = min(max_points_i, max(0, guide_quota))
        parts: list[Any] = []
        meta: dict[str, Any] = {
            "sampler": "yoloe_guided_component_adaptive",
            "frame_id": int(frame_id),
            "max_points": int(max_points_i),
            "guide_quota": int(guide_quota),
            "yoloe": yoloe_meta,
        }
        birth_interval = max(1, int(getattr(args, "edgetam_gap_birth_interval", 1)))
        warmup_frames = max(0, int(getattr(args, "edgetam_gap_birth_warmup_frames", 0)))
        force_area_ratio = max(0.0, float(getattr(args, "edgetam_gap_birth_force_guide_area_ratio", 0.0)))
        max_candidate_area = int(yoloe_meta.get("candidate_area_max", 0) or 0)
        h, w = mask_np.shape
        max_candidate_area_ratio = float(max_candidate_area) / float(max(h * w, 1))
        scheduled_birth = birth_interval <= 1 or gap_index % birth_interval == 0 or frame_pos <= warmup_frames
        force_large_birth = force_area_ratio > 0.0 and max_candidate_area_ratio >= force_area_ratio
        meta["birth_schedule"] = {
            "interval": int(birth_interval),
            "warmup_frames": int(warmup_frames),
            "scheduled": bool(scheduled_birth),
            "force_guide_area_ratio": float(force_area_ratio),
            "max_candidate_area": int(max_candidate_area),
            "max_candidate_area_ratio": float(max_candidate_area_ratio),
            "force_large_birth": bool(force_large_birth),
        }
        if not scheduled_birth and not force_large_birth:
            meta["skipped_by_birth_schedule"] = True
            meta["point_count"] = 0
            return torch.zeros((0, 2), device="cuda", dtype=torch.float32), meta

        if candidate_masks and guide_quota > 0:
            per_det = max(1, int(getattr(args, "yoloe_gap_points_per_detection", 1)))
            guided_parts: list[Any] = []
            guided_meta: list[dict[str, Any]] = []
            remaining_guide = int(guide_quota)
            for det_idx, candidate_mask in enumerate(candidate_masks):
                if remaining_guide <= 0:
                    break
                pts_g, meta_g = original_sampler(
                    candidate_mask.astype(bool),
                    max_points=int(min(per_det, remaining_guide)),
                    min_component_area=max(int(min_component_area), int(getattr(args, "yoloe_gap_min_guide_area", 600))),
                    base_points_per_component=1,
                    area_per_extra_point=max(int(area_per_extra_point), 1),
                    max_points_per_component=int(per_det),
                    seed=int(seed) + int(det_idx),
                )
                remaining_guide -= int(pts_g.shape[0])
                if int(pts_g.shape[0]) > 0:
                    guided_parts.append(pts_g)
                guided_meta.append(
                    {
                        "det_index": int(det_idx),
                        "area": int(np.count_nonzero(candidate_mask)),
                        "point_count": int(pts_g.shape[0]),
                    }
                )
            if guided_parts:
                pts_guided = torch.cat(guided_parts, dim=0)
                parts.append(pts_guided)
            meta["guided_sampling"] = {
                "mode": "per_yoloe_detection_mask",
                "candidate_mask_count": int(len(candidate_masks)),
                "points_per_detection": int(per_det),
                "point_count": int(sum(int(part.shape[0]) for part in guided_parts)),
                "per_detection": guided_meta[:32],
            }
        elif guide_mask is not None and guide_quota > 0:
            pts_g, meta_g = original_sampler(
                guide_mask.astype(bool),
                max_points=int(guide_quota),
                min_component_area=max(int(min_component_area), int(getattr(args, "yoloe_gap_min_guide_area", 600))),
                base_points_per_component=int(base_points_per_component),
                area_per_extra_point=int(area_per_extra_point),
                max_points_per_component=int(max_points_per_component),
                seed=int(seed),
            )
            if int(pts_g.shape[0]) > 0:
                parts.append(pts_g)
            meta["guided_sampling"] = meta_g

        remaining = max_points_i - int(sum(int(part.shape[0]) for part in parts))
        if remaining > 0 and bool(getattr(args, "yoloe_gap_fallback_to_uncovered", True)):
            fallback_mask = mask_np.astype(bool)
            if guide_mask is not None and bool(getattr(args, "yoloe_gap_exclude_guided_from_fallback", True)):
                fallback_mask = fallback_mask & ~guide_mask.astype(bool)
            pts_f, meta_f = original_sampler(
                fallback_mask,
                max_points=int(remaining),
                min_component_area=max(int(min_component_area), int(getattr(args, "yoloe_gap_fallback_min_component_area", min_component_area))),
                base_points_per_component=int(base_points_per_component),
                area_per_extra_point=int(area_per_extra_point),
                max_points_per_component=int(max_points_per_component),
                seed=int(seed) + 17,
            )
            if int(pts_f.shape[0]) > 0:
                parts.append(pts_f)
            meta["fallback_sampling"] = meta_f

        if parts:
            pts = torch.cat(parts, dim=0)[:max_points_i]
        else:
            pts = torch.zeros((0, 2), device="cuda", dtype=torch.float32)
        meta["point_count"] = int(pts.shape[0])
        return pts, meta

    return guided_sampler


def _make_gap_segment_wrapper(args: SimpleNamespace, original_segment: Any) -> Any:
    call_state = {"segment_call_index": 0}

    def wrapped_segment(*call_args: Any, **call_kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        call_index = int(call_state["segment_call_index"])
        call_state["segment_call_index"] = call_index + 1
        masks, stats = original_segment(*call_args, **call_kwargs)
        # The first call is frame-0 stage1 and the second call is frame-0
        # uncovered stage2 for this runner. Later calls are per-frame gap births.
        if call_index >= 2:
            filtered, filter_meta = _filter_gap_birth_masks(masks, args)
            stats = dict(stats)
            stats["edgetam_gap_birth_filter"] = filter_meta
            return filtered, stats
        return masks, stats

    return wrapped_segment


def run(args: SimpleNamespace) -> None:
    import torch

    reset_runtime_stats()
    setattr(add_masks_to_stream_state, "_args", args)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.empty((1,), device="cuda")
    original_setup_models = base.setup_models
    original_add_masks = base.add_masks_to_stream_state
    original_sampler = base.sample_component_adaptive_points_yx
    original_segment = base.run_sam2_point_segment_choice
    try:
        base.setup_models = setup_models
        base.add_masks_to_stream_state = add_masks_to_stream_state
        base.sample_component_adaptive_points_yx = _make_yoloe_guided_sampler(args, original_sampler)
        base.run_sam2_point_segment_choice = _make_gap_segment_wrapper(args, original_segment)
        base.run(args)
    finally:
        if hasattr(add_masks_to_stream_state, "_args"):
            delattr(add_masks_to_stream_state, "_args")
        base.setup_models = original_setup_models
        base.add_masks_to_stream_state = original_add_masks
        base.sample_component_adaptive_points_yx = original_sampler
        base.run_sam2_point_segment_choice = original_segment
