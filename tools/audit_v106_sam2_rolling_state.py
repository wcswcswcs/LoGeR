#!/usr/bin/env python3
"""SAM2-only v106 rolling-state adapter.

This adapter keeps the baseline SAM2 segmentation and tracking loop, but makes
the SAM2 video state append frames incrementally instead of initializing from a
prebuilt full video frame list. It installs local monkey patches on the created
predictor instance, leaving the upstream SAM2 source and existing v106 SAM2
runner untouched.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

import tools.audit_v105_baseline_x_sam2_twostage_tracking as base


REPO_ROOT = base.REPO_ROOT
load_config = base.load_config
make_args = base.make_args

_ROLLING_STATS: dict[str, Any] = {}


def reset_rolling_stats() -> None:
    global _ROLLING_STATS
    _ROLLING_STATS = {
        "schema_version": "stream4d_v106_sam2_rolling_state_stats_v1",
        "enabled": True,
        "state_init_mode": "sam2_init_state_video_path_none",
        "source_video_path_ignored": "",
        "add_frame_call_count": 0,
        "add_frame_runtime_sec": 0.0,
        "feature_cache_hit_count": 0,
        "feature_store_hit_count": 0,
        "feature_fallback_count": 0,
        "stream_add_masks_call_count": 0,
        "stream_add_masks_runtime_sec": 0.0,
        "stream_add_masks_input_mask_count": 0,
        "stream_add_masks_admitted_mask_count": 0,
        "stream_add_masks_skipped_mask_count": 0,
        "post_start_birth_filter_call_count": 0,
        "post_start_birth_filter_skipped_by_interval": 0,
        "post_start_birth_filter_skipped_by_area": 0,
        "post_start_birth_filter_skipped_by_max_area": 0,
        "post_start_birth_filter_skipped_by_uncovered_ratio": 0,
        "post_start_birth_filter_skipped_by_shape": 0,
        "post_start_birth_filter_skipped_by_bbox_frac": 0,
        "post_start_birth_filter_skipped_by_edge_touch": 0,
        "post_start_birth_filter_skipped_by_extent": 0,
        "post_start_birth_filter_skipped_by_core_area": 0,
        "post_start_birth_filter_skipped_by_cap": 0,
        "post_start_birth_filter_interval_rescue_frame_count": 0,
        "post_start_birth_filter_interval_rescue_mask_count": 0,
        "post_start_birth_filter_interval_rescue_by_visible_count": 0,
        "post_start_birth_filter_interval_rescue_by_foreground_ratio": 0,
        "post_start_birth_filter_immediate_area_admit_count": 0,
        "post_start_birth_filter_persistence_match_count": 0,
        "post_start_birth_filter_persistence_admit_count": 0,
        "post_start_birth_filter_persistence_limited_count": 0,
        "post_start_birth_filter_appearance_enabled_count": 0,
        "post_start_birth_filter_appearance_match_count": 0,
        "post_start_birth_filter_appearance_admit_count": 0,
        "post_start_birth_filter_pending_add_count": 0,
        "post_start_birth_filter_pending_update_count": 0,
        "post_start_birth_filter_pending_prune_count": 0,
        "post_start_birth_filter_pending_max_count": 0,
        "post_start_birth_filter_records": [],
        "birth_transaction_enabled_count": 0,
        "birth_transaction_queue_add_count": 0,
        "birth_transaction_queued_mask_count": 0,
        "birth_transaction_immediate_trigger_count": 0,
        "birth_transaction_min_pending_trigger_count": 0,
        "birth_transaction_min_total_area_trigger_count": 0,
        "birth_transaction_max_delay_trigger_count": 0,
        "birth_transaction_last_frame_trigger_count": 0,
        "birth_transaction_commit_count": 0,
        "birth_transaction_committed_mask_count": 0,
        "birth_transaction_commit_runtime_sec": 0.0,
        "birth_transaction_reconsolidate_call_count": 0,
        "birth_transaction_delay_frame_sum": 0,
        "birth_transaction_max_delay_frames_observed": 0,
        "birth_transaction_max_queue_mask_count": 0,
        "birth_transaction_max_queue_frame_count": 0,
        "birth_transaction_records": [],
        "reconsolidate_call_count": 0,
        "reconsolidate_runtime_sec": 0.0,
        "reconsolidate_frame_output_count_sum": 0,
        "birth_recon_preprune_call_count": 0,
        "birth_recon_prepruned_cond_frame_count": 0,
        "birth_recon_prepruned_noncond_frame_count": 0,
        "birth_recon_preprune_keep_frames": 0,
        "visual_export_skip_requested": False,
        "visual_export_lean_requested": False,
        "visual_export_label_only_requested": False,
        "visual_export_noop_save_count": 0,
        "visual_export_noop_imwrite_count": 0,
        "visual_export_noop_video_count": 0,
        "visual_export_noop_sheet_count": 0,
        "rolling_prune_call_count": 0,
        "rolling_pruned_image_count": 0,
        "rolling_pruned_cached_feature_count": 0,
        "stream_growth_prune_enabled_count": 0,
        "stream_growth_prune_call_count": 0,
        "stream_growth_pruned_object_count": 0,
        "stream_growth_prune_skipped_by_history_median_count": 0,
        "stream_growth_prune_history_reset_count": 0,
        "stream_growth_prune_records": [],
        "gap_output_filter_call_count": 0,
        "gap_output_filter_active_call_count": 0,
        "gap_output_filter_input_mask_count": 0,
        "gap_output_filter_kept_mask_count": 0,
        "gap_output_filter_dropped_mask_count": 0,
        "gap_output_filter_dropped_by_bbox_frac": 0,
        "gap_output_filter_dropped_by_edge_touch": 0,
        "gap_output_filter_dropped_by_extent": 0,
        "gap_output_filter_dropped_by_core_area": 0,
        "gap_birth_disabled_count": 0,
        "gap_birth_disabled_mask_count": 0,
        "preflight_consistency_repair_call_count": 0,
        "preflight_consistency_repair_missing_input_frame_count": 0,
        "preflight_consistency_repair_extra_consolidated_frame_count": 0,
        "preflight_consistency_repair_records": [],
        "max_rolling_image_count": 0,
        "max_cached_feature_count": 0,
        "max_num_frames_value": 0,
        "records": [],
    }


def get_rolling_stats() -> dict[str, Any]:
    if not _ROLLING_STATS:
        reset_rolling_stats()
    return deepcopy(_ROLLING_STATS)


def _record(kind: str, row: dict[str, Any]) -> None:
    if not _ROLLING_STATS:
        reset_rolling_stats()
    rows = _ROLLING_STATS.setdefault("records", [])
    if len(rows) < 512:
        rows.append({"kind": str(kind), **row})


def _record_birth_filter(row: dict[str, Any]) -> None:
    rows = _ROLLING_STATS.setdefault("post_start_birth_filter_records", [])
    if len(rows) < 512:
        rows.append(row)


def _record_birth_transaction(row: dict[str, Any]) -> None:
    rows = _ROLLING_STATS.setdefault("birth_transaction_records", [])
    if len(rows) < 512:
        rows.append(row)


def _record_growth_prune(row: dict[str, Any]) -> None:
    rows = _ROLLING_STATS.setdefault("stream_growth_prune_records", [])
    if len(rows) < 512:
        rows.append(row)


def _record_preflight_consistency_repair(row: dict[str, Any]) -> None:
    rows = _ROLLING_STATS.setdefault("preflight_consistency_repair_records", [])
    if len(rows) < 512:
        rows.append(row)


def _autocast_for_predictor(predictor: Any, device: Any):
    import torch
    from contextlib import nullcontext

    dtype = torch.float32
    try:
        dtype = next(predictor.parameters()).dtype
    except Exception:
        pass
    if getattr(device, "type", str(device)) == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _install_rolling_feature_lookup(predictor: Any) -> None:
    original_get_image_feature = predictor._get_image_feature

    def rolling_get_image_feature(self: Any, inference_state: dict[str, Any], frame_idx: int, batch_size: int):
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is not None:
            _ROLLING_STATS["feature_cache_hit_count"] = int(_ROLLING_STATS["feature_cache_hit_count"]) + 1
            return original_get_image_feature(inference_state, frame_idx, batch_size)

        store = inference_state.get("rolling_images_by_idx")
        if isinstance(store, dict) and int(frame_idx) in store:
            import torch

            _ROLLING_STATS["feature_store_hit_count"] = int(_ROLLING_STATS["feature_store_hit_count"]) + 1
            device = inference_state["device"]
            image = store[int(frame_idx)].to(device, non_blocking=True).float().unsqueeze(0)
            with torch.inference_mode(), _autocast_for_predictor(self, device):
                backbone_out = self.forward_image(image)
            inference_state["cached_features"][int(frame_idx)] = (image, backbone_out)
            return original_get_image_feature(inference_state, frame_idx, batch_size)

        _ROLLING_STATS["feature_fallback_count"] = int(_ROLLING_STATS["feature_fallback_count"]) + 1
        return original_get_image_feature(inference_state, frame_idx, batch_size)

    predictor._v106_original_get_image_feature = original_get_image_feature
    predictor._get_image_feature = MethodType(rolling_get_image_feature, predictor)


def _install_rolling_init_state(predictor: Any) -> None:
    original_init_state = predictor.init_state

    def rolling_init_state(
        self: Any,
        video_path: str | None = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> dict[str, Any]:
        if video_path is not None:
            _ROLLING_STATS["source_video_path_ignored"] = str(video_path)
        state = original_init_state(
            video_path=None,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=async_loading_frames,
        )
        state["v106_rolling_state_enabled"] = True
        state["rolling_images_by_idx"] = {}
        state["rolling_added_frame_indices"] = set()
        state["rolling_prune_events"] = []
        return state

    predictor._v106_original_init_state = original_init_state
    predictor.init_state = MethodType(rolling_init_state, predictor)


def _input_frame_indices(state: dict[str, Any]) -> set[int]:
    input_frames: set[int] = set()
    for per_obj in state.get("point_inputs_per_obj", {}).values():
        input_frames.update(int(v) for v in per_obj.keys())
    for per_obj in state.get("mask_inputs_per_obj", {}).values():
        input_frames.update(int(v) for v in per_obj.keys())
    return input_frames


def _consolidated_frame_indices(state: dict[str, Any]) -> set[int]:
    consolidated = state.get("consolidated_frame_inds", {})
    frames: set[int] = set()
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        frames.update(int(v) for v in consolidated.get(storage_key, set()))
    return frames


def _drop_state_frame_outputs(state: dict[str, Any], frame_idx: int) -> None:
    frame_idx_i = int(frame_idx)
    output_dict = state.get("output_dict", {})
    consolidated = state.get("consolidated_frame_inds", {})
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        output_dict.get(storage_key, {}).pop(frame_idx_i, None)
        consolidated.get(storage_key, set()).discard(frame_idx_i)
        for obj_output in state.get("output_dict_per_obj", {}).values():
            obj_output.get(storage_key, {}).pop(frame_idx_i, None)
        for obj_output in state.get("temp_output_dict_per_obj", {}).values():
            obj_output.get(storage_key, {}).pop(frame_idx_i, None)
    state.get("frames_already_tracked", {}).pop(frame_idx_i, None)


def _drop_state_frame_inputs(state: dict[str, Any], frame_idx: int) -> None:
    frame_idx_i = int(frame_idx)
    for per_obj in state.get("point_inputs_per_obj", {}).values():
        per_obj.pop(frame_idx_i, None)
    for per_obj in state.get("mask_inputs_per_obj", {}).values():
        per_obj.pop(frame_idx_i, None)


def _repair_preflight_frame_consistency(state: dict[str, Any]) -> dict[str, Any]:
    input_frames = _input_frame_indices(state)
    consolidated_frames = _consolidated_frame_indices(state)
    missing_consolidated = sorted(int(v) for v in input_frames - consolidated_frames)
    extra_consolidated = sorted(int(v) for v in consolidated_frames - input_frames)
    for frame_idx in extra_consolidated:
        _drop_state_frame_outputs(state, frame_idx)
    for frame_idx in missing_consolidated:
        _drop_state_frame_inputs(state, frame_idx)
        _drop_state_frame_outputs(state, frame_idx)
    after_input_frames = _input_frame_indices(state)
    after_consolidated_frames = _consolidated_frame_indices(state)
    event = {
        "missing_consolidated_frame_indices": missing_consolidated,
        "extra_consolidated_frame_indices": extra_consolidated,
        "after_missing_consolidated_frame_indices": sorted(
            int(v) for v in after_input_frames - after_consolidated_frames
        ),
        "after_extra_consolidated_frame_indices": sorted(
            int(v) for v in after_consolidated_frames - after_input_frames
        ),
    }
    if missing_consolidated or extra_consolidated:
        _ROLLING_STATS["preflight_consistency_repair_call_count"] = int(
            _ROLLING_STATS["preflight_consistency_repair_call_count"]
        ) + 1
        _ROLLING_STATS["preflight_consistency_repair_missing_input_frame_count"] = int(
            _ROLLING_STATS["preflight_consistency_repair_missing_input_frame_count"]
        ) + int(len(missing_consolidated))
        _ROLLING_STATS["preflight_consistency_repair_extra_consolidated_frame_count"] = int(
            _ROLLING_STATS["preflight_consistency_repair_extra_consolidated_frame_count"]
        ) + int(len(extra_consolidated))
        _record_preflight_consistency_repair(event)
        _record("preflight_consistency_repair", event)
    return event


def _install_preflight_consistency_repair(predictor: Any) -> None:
    if not hasattr(predictor, "propagate_in_video_preflight"):
        return
    original_preflight = predictor.propagate_in_video_preflight

    def repaired_preflight(self: Any, inference_state: dict[str, Any]):
        try:
            return original_preflight(inference_state)
        except AssertionError:
            event = _repair_preflight_frame_consistency(inference_state)
            if (
                event.get("after_missing_consolidated_frame_indices")
                or event.get("after_extra_consolidated_frame_indices")
            ):
                raise
            return original_preflight(inference_state)

    predictor._v106_original_propagate_in_video_preflight = original_preflight
    predictor.propagate_in_video_preflight = MethodType(repaired_preflight, predictor)


def install_rolling_state_support(predictor: Any) -> None:
    if getattr(predictor, "_v106_rolling_state_installed", False):
        return
    _install_rolling_feature_lookup(predictor)
    _install_rolling_init_state(predictor)
    _install_preflight_consistency_repair(predictor)
    predictor._v106_rolling_state_installed = True


def _rolling_add_frame(
    predictor: Any,
    state: dict[str, Any],
    *,
    frame_idx: int,
    rgb: np.ndarray,
) -> None:
    if int(frame_idx) in state.get("rolling_added_frame_indices", set()):
        return
    import torch
    from sam2.utils.misc import process_stream_frame

    started = time.time()
    device = state["device"]
    img_tensor, orig_h, orig_w = process_stream_frame(
        img_array=rgb,
        image_size=int(predictor.image_size),
        offload_to_cpu=bool(state.get("offload_video_to_cpu", False)),
        compute_device=device,
    )
    if state.get("video_height") is None:
        state["video_height"] = int(orig_h)
        state["video_width"] = int(orig_w)
    state["rolling_images_by_idx"][int(frame_idx)] = img_tensor
    state["rolling_added_frame_indices"].add(int(frame_idx))
    state["num_frames"] = max(int(state.get("num_frames", 0)), int(frame_idx) + 1)
    image_batch = img_tensor.to(device, non_blocking=True).float().unsqueeze(0)
    with torch.inference_mode(), _autocast_for_predictor(predictor, device):
        backbone_out = predictor.forward_image(image_batch)
    state["cached_features"][int(frame_idx)] = (image_batch, backbone_out)

    elapsed = float(time.time() - started)
    stats = _ROLLING_STATS
    stats["add_frame_call_count"] = int(stats["add_frame_call_count"]) + 1
    stats["add_frame_runtime_sec"] = float(stats["add_frame_runtime_sec"]) + elapsed
    stats["max_rolling_image_count"] = max(
        int(stats["max_rolling_image_count"]), int(len(state.get("rolling_images_by_idx", {})))
    )
    stats["max_cached_feature_count"] = max(
        int(stats["max_cached_feature_count"]), int(len(state.get("cached_features", {})))
    )
    stats["max_num_frames_value"] = max(int(stats["max_num_frames_value"]), int(state.get("num_frames", 0)))
    _record(
        "add_frame",
        {
            "frame_idx": int(frame_idx),
            "runtime_sec": elapsed,
            "rolling_image_count": int(len(state.get("rolling_images_by_idx", {}))),
            "cached_feature_count": int(len(state.get("cached_features", {}))),
            "num_frames": int(state.get("num_frames", 0)),
        },
    )


def _collect_required_image_indices(state: dict[str, Any], current_frame_idx: int) -> set[int]:
    keep: set[int] = {int(current_frame_idx)}
    output_dict = state.get("output_dict", {})
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        keep.update(int(v) for v in output_dict.get(storage_key, {}).keys())
    for frame_set in state.get("consolidated_frame_inds", {}).values():
        keep.update(int(v) for v in frame_set)
    for per_obj in state.get("point_inputs_per_obj", {}).values():
        keep.update(int(v) for v in per_obj.keys())
    for per_obj in state.get("mask_inputs_per_obj", {}).values():
        keep.update(int(v) for v in per_obj.keys())
    for row in state.get("v107_birth_transaction_queue", []):
        if isinstance(row, dict) and "frame_idx" in row:
            keep.add(int(row.get("frame_idx", current_frame_idx)))
    return keep


def _prune_rolling_frame_store(state: dict[str, Any], *, current_frame_idx: int) -> list[int]:
    store = state.get("rolling_images_by_idx")
    if not isinstance(store, dict):
        return []
    keep = _collect_required_image_indices(state, int(current_frame_idx))
    before_images = set(int(v) for v in store.keys())
    before_cache = set(int(v) for v in state.get("cached_features", {}).keys())
    for frame_idx in sorted(before_images - keep):
        store.pop(int(frame_idx), None)
    for frame_idx in sorted(before_cache - keep):
        state.get("cached_features", {}).pop(int(frame_idx), None)
    pruned_images = sorted(before_images - set(int(v) for v in store.keys()))
    pruned_cache = sorted(before_cache - set(int(v) for v in state.get("cached_features", {}).keys()))
    if pruned_images or pruned_cache:
        event = {
            "current_frame_idx": int(current_frame_idx),
            "pruned_image_indices": [int(v) for v in pruned_images],
            "pruned_cached_feature_indices": [int(v) for v in pruned_cache],
            "kept_image_count": int(len(store)),
            "kept_cached_feature_count": int(len(state.get("cached_features", {}))),
        }
        state.setdefault("rolling_prune_events", []).append(event)
        stats = _ROLLING_STATS
        stats["rolling_prune_call_count"] = int(stats["rolling_prune_call_count"]) + 1
        stats["rolling_pruned_image_count"] = int(stats["rolling_pruned_image_count"]) + int(len(pruned_images))
        stats["rolling_pruned_cached_feature_count"] = int(stats["rolling_pruned_cached_feature_count"]) + int(len(pruned_cache))
        _record("prune", event)
    _ROLLING_STATS["max_rolling_image_count"] = max(
        int(_ROLLING_STATS["max_rolling_image_count"]), int(len(store))
    )
    _ROLLING_STATS["max_cached_feature_count"] = max(
        int(_ROLLING_STATS["max_cached_feature_count"]), int(len(state.get("cached_features", {})))
    )
    return [int(v) for v in pruned_images]


def _prune_stream_outputs_for_birth_recon(
    state: dict[str, Any],
    *,
    current_frame_idx: int,
    keep_recent_frames: int,
    keep_frame_zero: bool = True,
) -> dict[str, list[int]]:
    keep_recent = int(keep_recent_frames)
    if keep_recent <= 0:
        return {"cond_frame_outputs": [], "non_cond_frame_outputs": []}
    min_keep_frame = int(current_frame_idx) - keep_recent + 1
    output_dict = state.get("output_dict", {})
    per_obj = state.get("output_dict_per_obj", {})
    temp_per_obj = state.get("temp_output_dict_per_obj", {})
    consolidated = state.get("consolidated_frame_inds", {})
    frames_already_tracked = state.get("frames_already_tracked", {})
    pruned: dict[str, list[int]] = {}
    for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
        outputs = output_dict.get(storage_key, {})
        remove_frames: list[int] = []
        for frame_idx in list(outputs.keys()):
            frame_idx_i = int(frame_idx)
            if keep_frame_zero and frame_idx_i == 0:
                continue
            if frame_idx_i < min_keep_frame:
                remove_frames.append(frame_idx_i)
        if storage_key == "cond_frame_outputs" and remove_frames:
            remaining_count = int(len(outputs)) - int(len(remove_frames))
            if remaining_count <= 0:
                remove_frames.remove(max(remove_frames))
        for frame_idx_i in sorted(remove_frames):
            outputs.pop(frame_idx_i, None)
            consolidated.get(storage_key, set()).discard(frame_idx_i)
            frames_already_tracked.pop(frame_idx_i, None)
            for obj_idx, obj_output in per_obj.items():
                obj_output.get(storage_key, {}).pop(frame_idx_i, None)
                state.get("point_inputs_per_obj", {}).get(obj_idx, {}).pop(frame_idx_i, None)
                state.get("mask_inputs_per_obj", {}).get(obj_idx, {}).pop(frame_idx_i, None)
            for obj_output in temp_per_obj.values():
                obj_output.get(storage_key, {}).pop(frame_idx_i, None)
        pruned[storage_key] = sorted(int(v) for v in remove_frames)
    cond_count = int(len(pruned.get("cond_frame_outputs", [])))
    noncond_count = int(len(pruned.get("non_cond_frame_outputs", [])))
    if cond_count or noncond_count:
        _ROLLING_STATS["birth_recon_preprune_call_count"] = int(
            _ROLLING_STATS["birth_recon_preprune_call_count"]
        ) + 1
        _ROLLING_STATS["birth_recon_prepruned_cond_frame_count"] = int(
            _ROLLING_STATS["birth_recon_prepruned_cond_frame_count"]
        ) + cond_count
        _ROLLING_STATS["birth_recon_prepruned_noncond_frame_count"] = int(
            _ROLLING_STATS["birth_recon_prepruned_noncond_frame_count"]
        ) + noncond_count
        _ROLLING_STATS["birth_recon_preprune_keep_frames"] = int(keep_recent)
        _record(
            "birth_recon_preprune",
            {
                "current_frame_idx": int(current_frame_idx),
                "keep_recent_frames": int(keep_recent),
                "pruned_cond_frame_indices": pruned.get("cond_frame_outputs", []),
                "pruned_noncond_frame_indices": pruned.get("non_cond_frame_outputs", []),
            },
        )
    return pruned


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool, copy=False)
    b = mask_b.astype(bool, copy=False)
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _prune_pending_births(state: dict[str, Any], *, frame_idx: int, ttl: int) -> list[dict[str, Any]]:
    if int(ttl) <= 0:
        state.pop("v106_pending_births", None)
        return []
    pending = state.setdefault("v106_pending_births", [])
    if not isinstance(pending, list):
        pending = []
    kept: list[dict[str, Any]] = []
    pruned = 0
    for row in pending:
        try:
            last_frame_idx = int(row.get("last_frame_idx", row.get("first_frame_idx", frame_idx)))
        except Exception:
            pruned += 1
            continue
        if int(frame_idx) - last_frame_idx <= int(ttl):
            kept.append(row)
        else:
            pruned += 1
    state["v106_pending_births"] = kept
    if pruned:
        _ROLLING_STATS["post_start_birth_filter_pending_prune_count"] = int(
            _ROLLING_STATS["post_start_birth_filter_pending_prune_count"]
        ) + int(pruned)
    return kept


def _find_best_pending_birth(
    pending: list[dict[str, Any]],
    mask: np.ndarray,
    *,
    min_iou: float,
) -> tuple[int | None, float]:
    best_idx: int | None = None
    best_iou = 0.0
    for pending_idx, row in enumerate(pending):
        pending_mask = row.get("mask")
        if pending_mask is None:
            continue
        iou = _mask_iou(mask, pending_mask)
        if iou > best_iou:
            best_idx = int(pending_idx)
            best_iou = float(iou)
    if best_idx is None or best_iou < float(min_iou):
        return None, float(best_iou)
    return best_idx, float(best_iou)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask.astype(bool, copy=False))
    if ys.size == 0:
        return -1.0, -1.0
    return float(ys.mean()), float(xs.mean())


def _mask_rgb_mean(rgb: np.ndarray | None, mask: np.ndarray) -> list[float] | None:
    if rgb is None:
        return None
    selected = rgb[mask.astype(bool, copy=False)]
    if selected.size == 0:
        return None
    return [float(v) for v in selected.astype(np.float32).mean(axis=0).tolist()]


def _appearance_feature(rgb: np.ndarray | None, mask: np.ndarray, area: int) -> dict[str, Any]:
    cy, cx = _mask_centroid(mask)
    return {
        "rgb_mean": _mask_rgb_mean(rgb, mask),
        "centroid_y": float(cy),
        "centroid_x": float(cx),
        "area": int(area),
    }


def _mask_shape_features(mask: np.ndarray) -> dict[str, Any]:
    mask_b = np.asarray(mask).astype(bool)
    h, w = mask_b.shape[:2]
    image_area = max(1, int(h) * int(w))
    area = int(np.count_nonzero(mask_b))
    if area <= 0:
        return {
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
    dist = base.cv2.distanceTransform(mask_b.astype(np.uint8), base.cv2.DIST_L2, 3)
    return {
        "bbox_area_frac": float((bw * bh) / image_area),
        "edge_touch_count": int(edge_touch_count),
        "extent": float(area / max(1, bw * bh)),
        "core16_area_px": int(np.count_nonzero(dist >= 16.0)),
    }


def _filter_masks_by_shape(
    masks: np.ndarray,
    *,
    current_uncovered_ratio: float,
    min_uncovered_ratio: float,
    max_bbox_frac: float,
    max_edge_touch_count: int,
    min_extent: float,
    min_core_area: int,
    min_input_mask_count: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    masks_b = np.asarray(masks).astype(bool)
    input_count = int(masks_b.shape[0]) if masks_b.ndim >= 3 else 0
    max_bbox_frac_threshold = max(0.0, float(max_bbox_frac))
    max_edge_touch_count_threshold = int(max_edge_touch_count)
    min_extent_threshold = max(0.0, float(min_extent))
    min_core_area_threshold = max(0, int(min_core_area))
    min_uncovered_ratio_threshold = max(0.0, float(min_uncovered_ratio))
    min_input_mask_count_threshold = max(0, int(min_input_mask_count))
    enabled = bool(
        max_bbox_frac_threshold > 0.0
        or max_edge_touch_count_threshold >= 0
        or min_extent_threshold > 0.0
        or min_core_area_threshold > 0
    )
    active = bool(
        enabled
        and float(current_uncovered_ratio) >= min_uncovered_ratio_threshold
        and input_count >= min_input_mask_count_threshold
    )
    stats: dict[str, Any] = {
        "enabled": bool(enabled),
        "active": bool(active),
        "input_mask_count": int(input_count),
        "kept_mask_count": int(input_count),
        "dropped_mask_count": 0,
        "current_uncovered_ratio": float(current_uncovered_ratio),
        "min_uncovered_ratio": float(min_uncovered_ratio_threshold),
        "min_input_mask_count": int(min_input_mask_count_threshold),
        "max_bbox_frac": float(max_bbox_frac_threshold),
        "max_edge_touch_count": int(max_edge_touch_count_threshold),
        "min_extent": float(min_extent_threshold),
        "min_core_area": int(min_core_area_threshold),
        "input_bbox_fracs": [],
        "input_edge_touch_counts": [],
        "input_extents": [],
        "input_core16_areas": [],
        "dropped_indices": [],
        "dropped_by_bbox_frac_indices": [],
        "dropped_by_edge_touch_indices": [],
        "dropped_by_extent_indices": [],
        "dropped_by_core_area_indices": [],
    }
    if input_count <= 0:
        return masks_b, stats
    shape_features = [_mask_shape_features(mask) for mask in masks_b]
    bbox_fracs = np.asarray([float(row["bbox_area_frac"]) for row in shape_features], dtype=np.float64)
    edge_touch_counts = np.asarray([int(row["edge_touch_count"]) for row in shape_features], dtype=np.int64)
    extents = np.asarray([float(row["extent"]) for row in shape_features], dtype=np.float64)
    core16_areas = np.asarray([int(row["core16_area_px"]) for row in shape_features], dtype=np.int64)
    stats["input_bbox_fracs"] = [float(v) for v in bbox_fracs.tolist()]
    stats["input_edge_touch_counts"] = [int(v) for v in edge_touch_counts.tolist()]
    stats["input_extents"] = [float(v) for v in extents.tolist()]
    stats["input_core16_areas"] = [int(v) for v in core16_areas.tolist()]
    if not active:
        return masks_b, stats

    keep = np.ones((input_count,), dtype=bool)
    bbox_reject = np.zeros_like(keep)
    if max_bbox_frac_threshold > 0.0:
        bbox_reject = bbox_fracs > max_bbox_frac_threshold
        keep &= ~bbox_reject
    edge_reject = np.zeros_like(keep)
    if max_edge_touch_count_threshold >= 0:
        edge_reject = keep & (edge_touch_counts > max_edge_touch_count_threshold)
        keep &= ~edge_reject
    extent_reject = np.zeros_like(keep)
    if min_extent_threshold > 0.0:
        extent_reject = keep & (extents < min_extent_threshold)
        keep &= ~extent_reject
    core_reject = np.zeros_like(keep)
    if min_core_area_threshold > 0:
        core_reject = keep & (core16_areas < min_core_area_threshold)
        keep &= ~core_reject
    dropped = ~keep
    stats.update(
        {
            "kept_mask_count": int(np.count_nonzero(keep)),
            "dropped_mask_count": int(np.count_nonzero(dropped)),
            "dropped_indices": [int(v) for v in np.flatnonzero(dropped).tolist()],
            "dropped_by_bbox_frac_indices": [int(v) for v in np.flatnonzero(bbox_reject).tolist()],
            "dropped_by_edge_touch_indices": [int(v) for v in np.flatnonzero(edge_reject).tolist()],
            "dropped_by_extent_indices": [int(v) for v in np.flatnonzero(extent_reject).tolist()],
            "dropped_by_core_area_indices": [int(v) for v in np.flatnonzero(core_reject).tolist()],
        }
    )
    return masks_b[keep], stats


def _appearance_distance(a: dict[str, Any], b: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    rgb_a = a.get("rgb_mean")
    rgb_b = b.get("rgb_mean")
    color_dist = None
    if rgb_a is not None and rgb_b is not None:
        color_dist = float(np.linalg.norm(np.asarray(rgb_a, dtype=np.float32) - np.asarray(rgb_b, dtype=np.float32)) / 255.0)
    ay, ax = float(a.get("centroid_y", -1.0)), float(a.get("centroid_x", -1.0))
    by, bx = float(b.get("centroid_y", -1.0)), float(b.get("centroid_x", -1.0))
    centroid_dist = None
    if ay >= 0.0 and ax >= 0.0 and by >= 0.0 and bx >= 0.0:
        centroid_dist = float(((ay - by) ** 2 + (ax - bx) ** 2) ** 0.5)
    area_a = max(1, int(a.get("area", 0)))
    area_b = max(1, int(b.get("area", 0)))
    area_ratio = float(max(area_a, area_b) / max(1, min(area_a, area_b)))
    return color_dist, centroid_dist, area_ratio


def _find_best_pending_birth_with_appearance(
    pending: list[dict[str, Any]],
    mask: np.ndarray,
    *,
    min_iou: float,
    feature: dict[str, Any] | None,
    appearance_enabled: bool,
    appearance_min_iou: float,
    appearance_max_color_distance: float,
    appearance_max_centroid_distance: float,
    appearance_max_area_ratio: float,
) -> tuple[int | None, float, str, dict[str, Any]]:
    best_idx, best_iou = _find_best_pending_birth(pending, mask, min_iou=min_iou)
    if best_idx is not None:
        return best_idx, float(best_iou), "iou", {"best_iou": float(best_iou)}
    if not bool(appearance_enabled) or feature is None:
        return None, float(best_iou), "none", {"best_iou": float(best_iou)}

    app_best_idx: int | None = None
    app_best_score = -1e9
    app_best_payload: dict[str, Any] = {"best_iou": float(best_iou)}
    for pending_idx, row in enumerate(pending):
        pending_mask = row.get("mask")
        if pending_mask is None:
            continue
        iou = _mask_iou(mask, pending_mask)
        pending_feature = row.get("appearance_feature")
        if not isinstance(pending_feature, dict):
            continue
        color_dist, centroid_dist, area_ratio = _appearance_distance(feature, pending_feature)
        if iou < float(appearance_min_iou):
            continue
        if color_dist is None or color_dist > float(appearance_max_color_distance):
            continue
        if centroid_dist is None or centroid_dist > float(appearance_max_centroid_distance):
            continue
        if area_ratio is None or area_ratio > float(appearance_max_area_ratio):
            continue
        centroid_term = centroid_dist / max(float(appearance_max_centroid_distance), 1.0)
        color_term = color_dist / max(float(appearance_max_color_distance), 1e-6)
        area_term = (area_ratio - 1.0) / max(float(appearance_max_area_ratio) - 1.0, 1e-6)
        score = float(iou) - float(color_term) - 0.35 * float(centroid_term) - 0.25 * float(area_term)
        if score > app_best_score:
            app_best_idx = int(pending_idx)
            app_best_score = float(score)
            app_best_payload = {
                "best_iou": float(iou),
                "appearance_score": float(score),
                "appearance_color_distance": float(color_dist),
                "appearance_centroid_distance": float(centroid_dist),
                "appearance_area_ratio": float(area_ratio),
            }
    if app_best_idx is None:
        return None, float(best_iou), "none", app_best_payload
    return app_best_idx, float(app_best_payload.get("best_iou", 0.0)), "appearance", app_best_payload


def _update_pending_birth(
    pending: list[dict[str, Any]],
    *,
    pending_idx: int | None,
    frame_idx: int,
    obj_id: int,
    mask: np.ndarray,
    area: int,
    hits: int,
    best_iou: float,
    appearance_feature: dict[str, Any] | None = None,
    match_kind: str = "none",
    match_payload: dict[str, Any] | None = None,
) -> None:
    row = {
        "obj_id": int(obj_id),
        "mask": mask.astype(bool, copy=True),
        "area": int(area),
        "first_frame_idx": int(frame_idx),
        "last_frame_idx": int(frame_idx),
        "hits": int(hits),
        "best_iou": float(best_iou),
        "match_kind": str(match_kind),
        "match_payload": match_payload or {},
    }
    if appearance_feature is not None:
        row["appearance_feature"] = appearance_feature
    if pending_idx is None:
        pending.append(row)
        _ROLLING_STATS["post_start_birth_filter_pending_add_count"] = int(
            _ROLLING_STATS["post_start_birth_filter_pending_add_count"]
        ) + 1
    else:
        previous = pending[int(pending_idx)]
        row["first_frame_idx"] = int(previous.get("first_frame_idx", frame_idx))
        pending[int(pending_idx)] = row
        _ROLLING_STATS["post_start_birth_filter_pending_update_count"] = int(
            _ROLLING_STATS["post_start_birth_filter_pending_update_count"]
        ) + 1
    _ROLLING_STATS["post_start_birth_filter_pending_max_count"] = max(
        int(_ROLLING_STATS["post_start_birth_filter_pending_max_count"]),
        int(len(pending)),
    )


def _filter_post_start_births(
    state: dict[str, Any],
    *,
    frame_idx: int,
    obj_ids: np.ndarray,
    masks: np.ndarray,
    rgb: np.ndarray | None = None,
    min_area: int,
    max_area: int,
    every: int,
    max_per_frame: int,
    max_uncovered_ratio: float = 0.0,
    max_bbox_frac: float = 0.0,
    max_edge_touch_count: int = -1,
    min_extent: float = 0.0,
    min_core_area: int = 0,
    shape_min_uncovered_ratio: float = 0.0,
    persistence_iou: float = 0.0,
    persistence_hits: int = 0,
    pending_ttl: int = 0,
    persistence_min_area: int = 0,
    persistence_max_per_frame: int = 0,
    immediate_area: int = 0,
    rescue_min_visible_count: int = 0,
    rescue_min_foreground_ratio: float = 0.0,
    appearance_enabled: bool = False,
    appearance_min_iou: float = 0.02,
    appearance_max_color_distance: float = 0.16,
    appearance_max_centroid_distance: float = 96.0,
    appearance_max_area_ratio: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    if masks.size == 0:
        return obj_ids, masks
    is_post_start = bool(state.get("tracking_has_started", False))
    existing = state.get("obj_id_to_idx", {})
    new_mask = np.asarray([int(obj_id) not in existing for obj_id in obj_ids.tolist()], dtype=bool)
    if not is_post_start or not bool(np.any(new_mask)):
        return obj_ids, masks

    _ROLLING_STATS["post_start_birth_filter_call_count"] = int(
        _ROLLING_STATS["post_start_birth_filter_call_count"]
    ) + 1
    areas = np.asarray([int(np.count_nonzero(mask)) for mask in masks.astype(bool)], dtype=np.int64)
    keep = np.ones((int(masks.shape[0]),), dtype=bool)

    interval = max(1, int(every))
    interval_allowed_by_cadence = (int(frame_idx) % interval) == 0
    current_filter_frame_idx = int(state.get("v106_current_frame_idx_for_birth_filter", -999999))
    if current_filter_frame_idx == int(frame_idx):
        last_visible_count = int(state.get("v106_current_visible_count_for_birth_filter", -1))
        last_foreground_ratio = float(state.get("v106_current_foreground_ratio_for_birth_filter", -1.0))
        last_infer_frame_idx = int(current_filter_frame_idx)
        rescue_signal_source = "post_disjoin_current_frame"
    else:
        last_visible_count = int(state.get("v106_last_infer_visible_count", -1))
        last_foreground_ratio = float(state.get("v106_last_infer_foreground_ratio", -1.0))
        last_infer_frame_idx = int(state.get("v106_last_infer_frame_idx", -999999))
        rescue_signal_source = "raw_infer_output"
    rescue_visible_threshold = max(0, int(rescue_min_visible_count))
    rescue_ratio_threshold = max(0.0, float(rescue_min_foreground_ratio))
    rescue_by_visible = (
        interval > 1
        and last_infer_frame_idx == int(frame_idx)
        and rescue_visible_threshold > 0
        and 0 <= last_visible_count < rescue_visible_threshold
    )
    rescue_by_ratio = (
        interval > 1
        and last_infer_frame_idx == int(frame_idx)
        and rescue_ratio_threshold > 0.0
        and 0.0 <= last_foreground_ratio < rescue_ratio_threshold
    )
    interval_rescue_allowed = bool(rescue_by_visible or rescue_by_ratio)
    interval_allowed = bool(interval_allowed_by_cadence or interval_rescue_allowed)
    area_threshold = max(0, int(min_area))
    area_ok = np.ones_like(new_mask, dtype=bool)
    if area_threshold > 0:
        area_ok = areas >= area_threshold
    max_area_threshold = max(0, int(max_area))
    max_area_ok = np.ones_like(new_mask, dtype=bool)
    if max_area_threshold > 0:
        max_area_ok = areas <= max_area_threshold
    max_uncovered_ratio_threshold = max(0.0, float(max_uncovered_ratio))
    current_foreground_ratio = max(0.0, min(1.0, float(last_foreground_ratio)))
    image_area = int(masks.shape[-2] * masks.shape[-1]) if masks.ndim >= 3 else 0
    current_uncovered_area = int(round(float(image_area) * (1.0 - current_foreground_ratio)))
    if current_uncovered_area <= 0:
        current_uncovered_area = int(image_area)
    uncovered_ratios = np.asarray(
        [float(area) / float(max(current_uncovered_area, 1)) for area in areas.tolist()],
        dtype=np.float64,
    )
    uncovered_ratio_ok = np.ones_like(new_mask, dtype=bool)
    if max_uncovered_ratio_threshold > 0.0:
        uncovered_ratio_ok = uncovered_ratios <= max_uncovered_ratio_threshold
    max_bbox_frac_threshold = max(0.0, float(max_bbox_frac))
    max_edge_touch_count_threshold = int(max_edge_touch_count)
    min_extent_threshold = max(0.0, float(min_extent))
    min_core_area_threshold = max(0, int(min_core_area))
    shape_min_uncovered_ratio_threshold = max(0.0, float(shape_min_uncovered_ratio))
    current_uncovered_ratio = float(current_uncovered_area) / float(max(image_area, 1))
    shape_filter_enabled = bool(
        max_bbox_frac_threshold > 0.0
        or max_edge_touch_count_threshold >= 0
        or min_extent_threshold > 0.0
        or min_core_area_threshold > 0
    )
    shape_filter_active = bool(
        shape_filter_enabled and current_uncovered_ratio >= shape_min_uncovered_ratio_threshold
    )
    shape_features = [_mask_shape_features(mask) for mask in masks.astype(bool)]
    bbox_fracs = np.asarray([float(row["bbox_area_frac"]) for row in shape_features], dtype=np.float64)
    edge_touch_counts = np.asarray([int(row["edge_touch_count"]) for row in shape_features], dtype=np.int64)
    extents = np.asarray([float(row["extent"]) for row in shape_features], dtype=np.float64)
    core16_areas = np.asarray([int(row["core16_area_px"]) for row in shape_features], dtype=np.int64)
    bbox_frac_ok = np.ones_like(new_mask, dtype=bool)
    if shape_filter_active and max_bbox_frac_threshold > 0.0:
        bbox_frac_ok = bbox_fracs <= max_bbox_frac_threshold
    edge_touch_ok = np.ones_like(new_mask, dtype=bool)
    if shape_filter_active and max_edge_touch_count_threshold >= 0:
        edge_touch_ok = edge_touch_counts <= max_edge_touch_count_threshold
    extent_ok = np.ones_like(new_mask, dtype=bool)
    if shape_filter_active and min_extent_threshold > 0.0:
        extent_ok = extents >= min_extent_threshold
    core_area_ok = np.ones_like(new_mask, dtype=bool)
    if shape_filter_active and min_core_area_threshold > 0:
        core_area_ok = core16_areas >= min_core_area_threshold
    shape_ok = bbox_frac_ok & edge_touch_ok & extent_ok & core_area_ok

    immediate_area_threshold = max(0, int(immediate_area))
    immediate_allowed = (
        new_mask & max_area_ok & uncovered_ratio_ok & shape_ok & (areas >= immediate_area_threshold)
        if immediate_area_threshold > 0
        else np.zeros_like(new_mask, dtype=bool)
    )

    persistence_threshold = float(persistence_iou)
    persistence_hit_threshold = max(0, int(persistence_hits))
    pending_ttl_i = max(0, int(pending_ttl))
    persistence_min_area_i = max(0, int(persistence_min_area))
    if persistence_min_area_i <= 0:
        persistence_min_area_i = area_threshold
    persistence_enabled = (
        pending_ttl_i > 0
        and persistence_threshold > 0.0
        and persistence_hit_threshold > 1
    )
    pending: list[dict[str, Any]] = (
        _prune_pending_births(state, frame_idx=int(frame_idx), ttl=pending_ttl_i)
        if persistence_enabled
        else []
    )
    if appearance_enabled and persistence_enabled:
        _ROLLING_STATS["post_start_birth_filter_appearance_enabled_count"] = int(
            _ROLLING_STATS["post_start_birth_filter_appearance_enabled_count"]
        ) + 1
    persistent_allowed = np.zeros_like(new_mask, dtype=bool)
    persistent_limited = np.zeros_like(new_mask, dtype=bool)
    persistent_hits_by_idx: dict[int, int] = {}
    persistent_iou_by_idx: dict[int, float] = {}
    matched_pending_by_idx: dict[int, int] = {}
    match_kind_by_idx: dict[int, str] = {}
    match_payload_by_idx: dict[int, dict[str, Any]] = {}
    appearance_feature_by_idx: dict[int, dict[str, Any]] = {}

    if persistence_enabled:
        for idx in np.flatnonzero(new_mask):
            idx_i = int(idx)
            if bool(immediate_allowed[idx_i]):
                continue
            if not bool(max_area_ok[idx_i]):
                continue
            if not bool(uncovered_ratio_ok[idx_i]):
                continue
            if not bool(shape_ok[idx_i]):
                continue
            if int(areas[idx_i]) < persistence_min_area_i:
                continue
            feature = _appearance_feature(rgb, masks[idx_i].astype(bool, copy=False), int(areas[idx_i]))
            appearance_feature_by_idx[idx_i] = feature
            best_pending_idx, best_iou, match_kind, match_payload = _find_best_pending_birth_with_appearance(
                pending,
                masks[idx_i].astype(bool, copy=False),
                min_iou=persistence_threshold,
                feature=feature,
                appearance_enabled=bool(appearance_enabled),
                appearance_min_iou=float(appearance_min_iou),
                appearance_max_color_distance=float(appearance_max_color_distance),
                appearance_max_centroid_distance=float(appearance_max_centroid_distance),
                appearance_max_area_ratio=float(appearance_max_area_ratio),
            )
            if best_pending_idx is None:
                continue
            previous_hits = int(pending[int(best_pending_idx)].get("hits", 1))
            hits = previous_hits + 1
            persistent_hits_by_idx[idx_i] = int(hits)
            persistent_iou_by_idx[idx_i] = float(best_iou)
            matched_pending_by_idx[idx_i] = int(best_pending_idx)
            match_kind_by_idx[idx_i] = str(match_kind)
            match_payload_by_idx[idx_i] = match_payload
            _ROLLING_STATS["post_start_birth_filter_persistence_match_count"] = int(
                _ROLLING_STATS["post_start_birth_filter_persistence_match_count"]
            ) + 1
            if match_kind == "appearance":
                _ROLLING_STATS["post_start_birth_filter_appearance_match_count"] = int(
                    _ROLLING_STATS["post_start_birth_filter_appearance_match_count"]
                ) + 1
            if hits >= persistence_hit_threshold:
                persistent_allowed[idx_i] = True
                if match_kind == "appearance":
                    _ROLLING_STATS["post_start_birth_filter_appearance_admit_count"] = int(
                        _ROLLING_STATS["post_start_birth_filter_appearance_admit_count"]
                    ) + 1

        persistence_frame_cap = max(0, int(persistence_max_per_frame))
        if persistence_frame_cap > 0 and int(np.count_nonzero(persistent_allowed)) > persistence_frame_cap:
            candidates = np.flatnonzero(persistent_allowed)
            order = candidates[np.argsort(-areas[candidates], kind="stable")]
            keep_persistent = np.zeros_like(persistent_allowed)
            keep_persistent[order[:persistence_frame_cap]] = True
            persistent_limited = persistent_allowed & ~keep_persistent
            persistent_allowed &= keep_persistent
            _ROLLING_STATS["post_start_birth_filter_persistence_limited_count"] = int(
                _ROLLING_STATS["post_start_birth_filter_persistence_limited_count"]
            ) + int(np.count_nonzero(persistent_limited))

    protected_allowed = immediate_allowed | persistent_allowed
    normal_candidates = new_mask & ~protected_allowed & area_ok & max_area_ok & uncovered_ratio_ok & shape_ok
    if interval > 1 and not interval_allowed:
        normal_candidates[:] = False

    cap = int(max_per_frame)
    cap_reject = np.zeros_like(new_mask, dtype=bool)
    normal_allowed = normal_candidates.copy()
    if cap > 0 and int(np.count_nonzero(normal_allowed)) > cap:
        candidates = np.flatnonzero(normal_allowed)
        order = candidates[np.argsort(-areas[candidates], kind="stable")]
        keep_cap = np.zeros_like(normal_allowed)
        keep_cap[order[:cap]] = True
        cap_reject = normal_allowed & ~keep_cap
        normal_allowed &= keep_cap

    keep_new = protected_allowed | normal_allowed
    interval_reject = (
        (new_mask & ~keep_new & ~protected_allowed)
        if interval > 1 and not interval_allowed
        else np.zeros_like(new_mask, dtype=bool)
    )
    area_reject = np.zeros_like(new_mask, dtype=bool)
    if area_threshold > 0:
        area_reject = new_mask & ~keep_new & ~protected_allowed & ~interval_reject & ~area_ok
    max_area_reject = np.zeros_like(new_mask, dtype=bool)
    if max_area_threshold > 0:
        max_area_reject = new_mask & ~keep_new & ~protected_allowed & ~interval_reject & area_ok & ~max_area_ok
    uncovered_ratio_reject = np.zeros_like(new_mask, dtype=bool)
    if max_uncovered_ratio_threshold > 0.0:
        uncovered_ratio_reject = (
            new_mask
            & ~keep_new
            & ~protected_allowed
            & ~interval_reject
            & area_ok
            & max_area_ok
            & ~uncovered_ratio_ok
        )
    shape_reject_base = (
        new_mask
        & ~keep_new
        & ~protected_allowed
        & ~interval_reject
        & area_ok
        & max_area_ok
        & uncovered_ratio_ok
    )
    bbox_frac_reject = np.zeros_like(new_mask, dtype=bool)
    if max_bbox_frac_threshold > 0.0:
        bbox_frac_reject = shape_reject_base & ~bbox_frac_ok
    edge_touch_reject = np.zeros_like(new_mask, dtype=bool)
    if max_edge_touch_count_threshold >= 0:
        edge_touch_reject = shape_reject_base & bbox_frac_ok & ~edge_touch_ok
    extent_reject = np.zeros_like(new_mask, dtype=bool)
    if min_extent_threshold > 0.0:
        extent_reject = shape_reject_base & bbox_frac_ok & edge_touch_ok & ~extent_ok
    core_area_reject = np.zeros_like(new_mask, dtype=bool)
    if min_core_area_threshold > 0:
        core_area_reject = shape_reject_base & bbox_frac_ok & edge_touch_ok & extent_ok & ~core_area_ok
    shape_reject = bbox_frac_reject | edge_touch_reject | extent_reject | core_area_reject
    cap_reject = cap_reject & ~keep_new
    _ROLLING_STATS["post_start_birth_filter_skipped_by_interval"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_interval"]
    ) + int(np.count_nonzero(interval_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_area"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_area"]
    ) + int(np.count_nonzero(area_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_max_area"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_max_area"]
    ) + int(np.count_nonzero(max_area_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_uncovered_ratio"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_uncovered_ratio"]
    ) + int(np.count_nonzero(uncovered_ratio_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_shape"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_shape"]
    ) + int(np.count_nonzero(shape_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_bbox_frac"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_bbox_frac"]
    ) + int(np.count_nonzero(bbox_frac_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_edge_touch"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_edge_touch"]
    ) + int(np.count_nonzero(edge_touch_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_extent"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_extent"]
    ) + int(np.count_nonzero(extent_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_core_area"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_core_area"]
    ) + int(np.count_nonzero(core_area_reject))
    _ROLLING_STATS["post_start_birth_filter_skipped_by_cap"] = int(
        _ROLLING_STATS["post_start_birth_filter_skipped_by_cap"]
    ) + int(np.count_nonzero(cap_reject))
    if interval_rescue_allowed:
        _ROLLING_STATS["post_start_birth_filter_interval_rescue_frame_count"] = int(
            _ROLLING_STATS["post_start_birth_filter_interval_rescue_frame_count"]
        ) + 1
        _ROLLING_STATS["post_start_birth_filter_interval_rescue_mask_count"] = int(
            _ROLLING_STATS["post_start_birth_filter_interval_rescue_mask_count"]
        ) + int(np.count_nonzero(normal_allowed))
        if rescue_by_visible:
            _ROLLING_STATS["post_start_birth_filter_interval_rescue_by_visible_count"] = int(
                _ROLLING_STATS["post_start_birth_filter_interval_rescue_by_visible_count"]
            ) + 1
        if rescue_by_ratio:
            _ROLLING_STATS["post_start_birth_filter_interval_rescue_by_foreground_ratio"] = int(
                _ROLLING_STATS["post_start_birth_filter_interval_rescue_by_foreground_ratio"]
            ) + 1
    _ROLLING_STATS["post_start_birth_filter_immediate_area_admit_count"] = int(
        _ROLLING_STATS["post_start_birth_filter_immediate_area_admit_count"]
    ) + int(np.count_nonzero(immediate_allowed))
    _ROLLING_STATS["post_start_birth_filter_persistence_admit_count"] = int(
        _ROLLING_STATS["post_start_birth_filter_persistence_admit_count"]
    ) + int(np.count_nonzero(persistent_allowed))

    if persistence_enabled:
        matched_pending_to_remove = {
            int(matched_pending_by_idx[int(idx)])
            for idx in np.flatnonzero(keep_new & persistent_allowed)
            if int(idx) in matched_pending_by_idx
        }
        for idx in np.flatnonzero(new_mask & ~keep_new):
            idx_i = int(idx)
            if int(areas[idx_i]) < persistence_min_area_i:
                continue
            if not bool(max_area_ok[idx_i]):
                continue
            if not bool(uncovered_ratio_ok[idx_i]):
                continue
            if not bool(shape_ok[idx_i]):
                continue
            pending_idx = matched_pending_by_idx.get(idx_i)
            if pending_idx is not None and int(pending_idx) in matched_pending_to_remove:
                pending_idx = None
            feature = appearance_feature_by_idx.get(idx_i)
            if feature is None:
                feature = _appearance_feature(rgb, masks[idx_i].astype(bool, copy=False), int(areas[idx_i]))
            _update_pending_birth(
                pending,
                pending_idx=pending_idx,
                frame_idx=int(frame_idx),
                obj_id=int(obj_ids[idx_i]),
                mask=masks[idx_i].astype(bool, copy=False),
                area=int(areas[idx_i]),
                hits=int(persistent_hits_by_idx.get(idx_i, 1)),
                best_iou=float(persistent_iou_by_idx.get(idx_i, 0.0)),
                appearance_feature=feature if bool(appearance_enabled) else None,
                match_kind=str(match_kind_by_idx.get(idx_i, "none")),
                match_payload=match_payload_by_idx.get(idx_i, {}),
            )
        if matched_pending_to_remove:
            state["v106_pending_births"] = [
                row for idx, row in enumerate(pending) if int(idx) not in matched_pending_to_remove
            ]
        else:
            state["v106_pending_births"] = pending
        _ROLLING_STATS["post_start_birth_filter_pending_max_count"] = max(
            int(_ROLLING_STATS["post_start_birth_filter_pending_max_count"]),
            int(len(state.get("v106_pending_births", []))),
        )

    keep[new_mask] = keep_new[new_mask]
    skipped = int(np.count_nonzero(~keep))
    admitted = int(np.count_nonzero(keep))
    _record_birth_filter(
        {
            "frame_idx": int(frame_idx),
            "input_mask_count": int(masks.shape[0]),
            "new_mask_count": int(np.count_nonzero(new_mask)),
            "admitted_mask_count": admitted,
            "skipped_mask_count": skipped,
            "min_area": int(area_threshold),
            "max_area": int(max_area_threshold),
            "max_uncovered_ratio": float(max_uncovered_ratio_threshold),
            "current_uncovered_area": int(current_uncovered_area),
            "current_uncovered_ratio": float(current_uncovered_ratio),
            "max_bbox_frac": float(max_bbox_frac_threshold),
            "max_edge_touch_count": int(max_edge_touch_count_threshold),
            "min_extent": float(min_extent_threshold),
            "min_core_area": int(min_core_area_threshold),
            "shape_min_uncovered_ratio": float(shape_min_uncovered_ratio_threshold),
            "shape_filter_enabled": bool(shape_filter_enabled),
            "shape_filter_active": bool(shape_filter_active),
            "every": int(interval),
            "interval_allowed_by_cadence": bool(interval_allowed_by_cadence),
            "interval_rescue_allowed": bool(interval_rescue_allowed),
            "interval_rescue_by_visible_count": bool(rescue_by_visible),
            "interval_rescue_by_foreground_ratio": bool(rescue_by_ratio),
            "last_infer_visible_count": int(last_visible_count),
            "last_infer_foreground_ratio": float(last_foreground_ratio),
            "rescue_signal_source": str(rescue_signal_source),
            "rescue_min_visible_count": int(rescue_visible_threshold),
            "rescue_min_foreground_ratio": float(rescue_ratio_threshold),
            "max_per_frame": int(cap),
            "persistence_iou": float(persistence_threshold),
            "persistence_hits": int(persistence_hit_threshold),
            "pending_ttl": int(pending_ttl_i),
            "persistence_min_area": int(persistence_min_area_i),
            "persistence_max_per_frame": int(persistence_max_per_frame),
            "immediate_area": int(immediate_area_threshold),
            "immediate_area_admit_count": int(np.count_nonzero(immediate_allowed)),
            "persistence_match_count": int(len(persistent_hits_by_idx)),
            "persistence_admit_count": int(np.count_nonzero(persistent_allowed)),
            "persistence_limited_count": int(np.count_nonzero(persistent_limited)),
            "appearance_enabled": bool(appearance_enabled and persistence_enabled),
            "appearance_min_iou": float(appearance_min_iou),
            "appearance_max_color_distance": float(appearance_max_color_distance),
            "appearance_max_centroid_distance": float(appearance_max_centroid_distance),
            "appearance_max_area_ratio": float(appearance_max_area_ratio),
            "appearance_match_count": int(
                sum(1 for value in match_kind_by_idx.values() if str(value) == "appearance")
            ),
            "appearance_admit_count": int(
                sum(
                    1
                    for idx in np.flatnonzero(persistent_allowed)
                    if str(match_kind_by_idx.get(int(idx), "")) == "appearance"
                )
            ),
            "match_kinds": {str(idx): str(value) for idx, value in match_kind_by_idx.items()},
            "pending_count_after": int(len(state.get("v106_pending_births", []))),
            "input_areas": [int(v) for v in areas.tolist()],
            "input_uncovered_ratios": [float(v) for v in uncovered_ratios.tolist()],
            "input_bbox_fracs": [float(v) for v in bbox_fracs.tolist()],
            "input_edge_touch_counts": [int(v) for v in edge_touch_counts.tolist()],
            "input_extents": [float(v) for v in extents.tolist()],
            "input_core16_areas": [int(v) for v in core16_areas.tolist()],
            "admitted_obj_ids": [int(v) for v in obj_ids[keep].tolist()],
            "skipped_obj_ids": [int(v) for v in obj_ids[~keep].tolist()],
            "max_area_skipped_obj_ids": [int(v) for v in obj_ids[max_area_reject].tolist()],
            "uncovered_ratio_skipped_obj_ids": [int(v) for v in obj_ids[uncovered_ratio_reject].tolist()],
            "shape_skipped_obj_ids": [int(v) for v in obj_ids[shape_reject].tolist()],
            "bbox_frac_skipped_obj_ids": [int(v) for v in obj_ids[bbox_frac_reject].tolist()],
            "edge_touch_skipped_obj_ids": [int(v) for v in obj_ids[edge_touch_reject].tolist()],
            "extent_skipped_obj_ids": [int(v) for v in obj_ids[extent_reject].tolist()],
            "core_area_skipped_obj_ids": [int(v) for v in obj_ids[core_area_reject].tolist()],
            "persistence_admitted_obj_ids": [int(v) for v in obj_ids[persistent_allowed].tolist()],
            "immediate_admitted_obj_ids": [int(v) for v in obj_ids[immediate_allowed].tolist()],
        }
    )
    return obj_ids[keep], masks[keep]


def run(args: SimpleNamespace) -> None:
    reset_rolling_stats()
    frame_ids = base.parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = (REPO_ROOT / args.rgb_root).resolve() if not Path(args.rgb_root).is_absolute() else Path(args.rgb_root)
    rgb_root = rgb_root / args.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    rgbs_by_idx: dict[int, np.ndarray] = {}

    def get_rgb(frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx not in rgbs_by_idx:
            rgbs_by_idx[frame_idx] = base.read_rgb(frame_paths[frame_idx])
        return rgbs_by_idx[frame_idx]

    original_setup_models = base.setup_models
    original_add_masks = base.add_masks_to_stream_state
    original_infer = base.infer_stream_frame
    original_prune_oversized = base.prune_stream_oversized_visible_objects
    original_prune_noncond = base.prune_stream_noncond_memory
    original_reconsolidate = base.reconsolidate_stream_state_outputs
    original_run_sam2_point_segment_choice = base.run_sam2_point_segment_choice
    original_overlay_label = base.overlay_label
    original_annotate_frame = base.annotate_frame
    original_make_sheet_grid = base.make_sheet_grid
    original_write_video = base.write_video
    original_imwrite = base.cv2.imwrite

    skip_visual_export = bool(getattr(args, "skip_visual_export", False))
    lean_visual_export = bool(getattr(args, "lean_visual_export", False))
    label_only_visual_export = bool(getattr(args, "label_only_visual_export", False))
    birth_min_area = int(getattr(args, "birth_admission_min_area", 0))
    birth_max_area = int(getattr(args, "birth_admission_max_area", 0))
    birth_max_uncovered_ratio = float(getattr(args, "birth_admission_max_uncovered_ratio", 0.0))
    birth_max_bbox_frac = float(getattr(args, "birth_admission_max_bbox_frac", 0.0))
    birth_max_edge_touch_count = int(getattr(args, "birth_admission_max_edge_touch_count", -1))
    birth_min_extent = float(getattr(args, "birth_admission_min_extent", 0.0))
    birth_min_core_area = int(getattr(args, "birth_admission_min_core_area_px", 0))
    birth_shape_min_uncovered_ratio = float(getattr(args, "birth_admission_shape_min_uncovered_ratio", 0.0))
    gap_output_max_bbox_frac = float(getattr(args, "gap_output_max_bbox_frac", 0.0))
    gap_output_max_edge_touch_count = int(getattr(args, "gap_output_max_edge_touch_count", -1))
    gap_output_min_extent = float(getattr(args, "gap_output_min_extent", 0.0))
    gap_output_min_core_area = int(getattr(args, "gap_output_min_core_area_px", 0))
    gap_output_shape_min_uncovered_ratio = float(getattr(args, "gap_output_shape_min_uncovered_ratio", 0.0))
    gap_output_min_input_mask_count = int(getattr(args, "gap_output_min_input_mask_count", 0))
    disable_gap_birth = bool(getattr(args, "disable_gap_birth", False))
    birth_every = int(getattr(args, "birth_admission_every", 1))
    birth_max_per_frame = int(getattr(args, "birth_admission_max_per_frame", 0))
    birth_persistence_iou = float(getattr(args, "birth_admission_persistence_iou", 0.0))
    birth_persistence_hits = int(getattr(args, "birth_admission_persistence_hits", 0))
    birth_pending_ttl = int(getattr(args, "birth_admission_pending_ttl", 0))
    birth_persistence_min_area = int(getattr(args, "birth_admission_persistence_min_area", 0))
    birth_persistence_max_per_frame = int(getattr(args, "birth_admission_persistence_max_per_frame", 0))
    birth_immediate_area = int(getattr(args, "birth_admission_immediate_area", 0))
    birth_rescue_min_visible_count = int(getattr(args, "birth_admission_rescue_min_visible_count", 0))
    birth_rescue_min_foreground_ratio = float(getattr(args, "birth_admission_rescue_min_foreground_ratio", 0.0))
    birth_recon_prune_keep_frames = int(getattr(args, "birth_recon_prune_keep_frames", 0))
    birth_appearance_enabled = bool(getattr(args, "birth_admission_appearance_enabled", False))
    birth_appearance_min_iou = float(getattr(args, "birth_admission_appearance_min_iou", 0.02))
    birth_appearance_max_color_distance = float(getattr(args, "birth_admission_appearance_max_color_distance", 0.16))
    birth_appearance_max_centroid_distance = float(getattr(args, "birth_admission_appearance_max_centroid_distance", 96.0))
    birth_appearance_max_area_ratio = float(getattr(args, "birth_admission_appearance_max_area_ratio", 4.0))
    birth_transaction_enabled = bool(getattr(args, "birth_transaction_enabled", False))
    birth_transaction_min_pending = int(getattr(args, "birth_transaction_min_pending", 0))
    birth_transaction_max_delay_frames = int(getattr(args, "birth_transaction_max_delay_frames", 0))
    birth_transaction_immediate_area = int(getattr(args, "birth_transaction_immediate_area", 0))
    birth_transaction_min_total_area = int(getattr(args, "birth_transaction_min_total_area", 0))
    growth_prune_ratio = max(0.0, float(getattr(args, "stream_growth_prune_ratio", 0.0)))
    growth_prune_min_area = max(0, int(getattr(args, "stream_growth_prune_min_area", 0)))
    growth_prune_history = max(1, int(getattr(args, "stream_growth_prune_history", 5)))
    growth_prune_warmup = max(1, int(getattr(args, "stream_growth_prune_warmup", 3)))
    growth_prune_max_history_median_area = max(
        0,
        int(getattr(args, "stream_growth_prune_max_history_median_area", 0)),
    )
    growth_prune_action = str(getattr(args, "stream_growth_prune_action", "prune") or "prune").strip().lower()
    if growth_prune_action not in {"prune", "alert_only", "suppress_output"}:
        raise ValueError(f"unsupported stream growth prune action: {growth_prune_action}")
    growth_prune_enabled = bool(growth_prune_ratio > 0.0 and growth_prune_min_area > 0)
    if growth_prune_enabled:
        _ROLLING_STATS["stream_growth_prune_enabled_count"] = int(
            _ROLLING_STATS["stream_growth_prune_enabled_count"]
        ) + 1
    growth_area_history_by_obj: dict[int, list[int]] = {}
    last_chunk_frame_idx = max(0, int(len(frame_ids)) - 1)
    _ROLLING_STATS["visual_export_skip_requested"] = bool(skip_visual_export)
    _ROLLING_STATS["visual_export_lean_requested"] = bool(lean_visual_export)
    _ROLLING_STATS["visual_export_label_only_requested"] = bool(label_only_visual_export)
    if birth_transaction_enabled:
        _ROLLING_STATS["birth_transaction_enabled_count"] = 1

    def setup_models_rolling(inner_args: SimpleNamespace) -> dict[str, Any]:
        models = original_setup_models(inner_args)
        install_rolling_state_support(models["tracker_model"])
        return models

    def reconsolidate_timed(predictor: Any, state: dict[str, Any]) -> None:
        started = time.time()
        output_dict = state.get("output_dict", {})
        frame_output_count = int(
            len(output_dict.get("cond_frame_outputs", {})) + len(output_dict.get("non_cond_frame_outputs", {}))
        )
        try:
            return original_reconsolidate(predictor, state)
        finally:
            elapsed = float(time.time() - started)
            _ROLLING_STATS["reconsolidate_call_count"] = int(_ROLLING_STATS["reconsolidate_call_count"]) + 1
            _ROLLING_STATS["reconsolidate_runtime_sec"] = float(_ROLLING_STATS["reconsolidate_runtime_sec"]) + elapsed
            _ROLLING_STATS["reconsolidate_frame_output_count_sum"] = int(
                _ROLLING_STATS["reconsolidate_frame_output_count_sum"]
            ) + frame_output_count
            _record(
                "reconsolidate",
                {
                    "runtime_sec": elapsed,
                    "frame_output_count": int(frame_output_count),
                    "object_count": int(len(state.get("obj_ids", []))),
                },
            )

    def birth_transaction_queue(state: dict[str, Any]) -> list[dict[str, Any]]:
        queue = state.setdefault("v107_birth_transaction_queue", [])
        if not isinstance(queue, list):
            queue = []
            state["v107_birth_transaction_queue"] = queue
        return queue

    def birth_transaction_queue_mask_count(state: dict[str, Any]) -> int:
        return int(sum(int(row.get("mask_count", 0)) for row in birth_transaction_queue(state)))

    def update_birth_transaction_highwater(state: dict[str, Any]) -> None:
        queue = birth_transaction_queue(state)
        mask_count = int(sum(int(row.get("mask_count", 0)) for row in queue))
        frame_count = int(len({int(row.get("frame_idx", -1)) for row in queue}))
        _ROLLING_STATS["birth_transaction_max_queue_mask_count"] = max(
            int(_ROLLING_STATS["birth_transaction_max_queue_mask_count"]), mask_count
        )
        _ROLLING_STATS["birth_transaction_max_queue_frame_count"] = max(
            int(_ROLLING_STATS["birth_transaction_max_queue_frame_count"]), frame_count
        )

    def enqueue_birth_transaction(
        state: dict[str, Any],
        *,
        frame_idx: int,
        tracker: str,
        obj_ids: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        if masks.size == 0:
            return
        bool_masks = masks.astype(bool, copy=True)
        areas = [int(np.count_nonzero(mask)) for mask in bool_masks]
        row = {
            "frame_idx": int(frame_idx),
            "queued_at_frame_idx": int(frame_idx),
            "tracker": str(tracker),
            "obj_ids": obj_ids.astype(np.int64, copy=True),
            "masks": bool_masks,
            "areas": areas,
            "area_sum": int(sum(areas)),
            "mask_count": int(bool_masks.shape[0]),
        }
        birth_transaction_queue(state).append(row)
        _ROLLING_STATS["birth_transaction_queue_add_count"] = int(
            _ROLLING_STATS["birth_transaction_queue_add_count"]
        ) + 1
        _ROLLING_STATS["birth_transaction_queued_mask_count"] = int(
            _ROLLING_STATS["birth_transaction_queued_mask_count"]
        ) + int(bool_masks.shape[0])
        update_birth_transaction_highwater(state)
        _record_birth_transaction(
            {
                "event": "enqueue",
                "frame_idx": int(frame_idx),
                "mask_count": int(bool_masks.shape[0]),
                "obj_ids": [int(v) for v in obj_ids.tolist()],
                "area_sum": int(sum(areas)),
            }
        )

    def birth_transaction_trigger(state: dict[str, Any], *, current_frame_idx: int) -> tuple[bool, str]:
        queue = birth_transaction_queue(state)
        mask_count = int(sum(int(row.get("mask_count", 0)) for row in queue))
        if mask_count <= 0:
            return False, "empty"
        total_area = int(sum(int(row.get("area_sum", 0)) for row in queue))
        oldest = min(int(row.get("queued_at_frame_idx", current_frame_idx)) for row in queue)
        if birth_transaction_min_pending > 0 and mask_count >= int(birth_transaction_min_pending):
            return True, "min_pending"
        if birth_transaction_min_total_area > 0 and total_area >= int(birth_transaction_min_total_area):
            return True, "min_total_area"
        if birth_transaction_max_delay_frames > 0 and int(current_frame_idx) - oldest >= int(
            birth_transaction_max_delay_frames
        ):
            return True, "max_delay"
        if int(current_frame_idx) >= int(last_chunk_frame_idx):
            return True, "last_frame"
        return False, "not_triggered"

    def commit_birth_transaction(
        predictor: Any,
        state: dict[str, Any],
        *,
        current_frame_idx: int,
        reason: str,
    ) -> int:
        import torch

        queue = birth_transaction_queue(state)
        if not queue:
            return 0
        started = time.time()
        old_obj_count = int(len(state.get("obj_ids", [])))
        old_tracking_started = bool(state.get("tracking_has_started", False))
        min_frame_idx = min(int(row.get("frame_idx", current_frame_idx)) for row in queue)
        for frame_idx_i in sorted({int(row.get("frame_idx", current_frame_idx)) for row in queue}):
            if state.get("v106_rolling_state_enabled"):
                _rolling_add_frame(predictor, state, frame_idx=frame_idx_i, rgb=get_rgb(frame_idx_i))
        if state.get("v106_rolling_state_enabled") and birth_recon_prune_keep_frames > 0:
            _prune_stream_outputs_for_birth_recon(
                state,
                current_frame_idx=int(current_frame_idx),
                keep_recent_frames=int(birth_recon_prune_keep_frames),
            )
            _prune_rolling_frame_store(state, current_frame_idx=int(current_frame_idx))
        frames_already_tracked = state.get("frames_already_tracked", {})
        if isinstance(frames_already_tracked, dict):
            for tracked_frame_idx in list(frames_already_tracked.keys()):
                if int(tracked_frame_idx) >= int(min_frame_idx):
                    frames_already_tracked.pop(tracked_frame_idx, None)
        committed = 0
        duplicate_skipped = 0
        delays: list[int] = []
        state["tracking_has_started"] = False
        try:
            for row in queue:
                frame_idx_i = int(row.get("frame_idx", current_frame_idx))
                tracker_i = str(row.get("tracker", "sam2"))
                obj_ids_i = row.get("obj_ids")
                masks_i = row.get("masks")
                if obj_ids_i is None or masks_i is None:
                    continue
                for obj_id, mask in zip(np.asarray(obj_ids_i).tolist(), np.asarray(masks_i).astype(bool), strict=False):
                    if int(obj_id) in state.get("obj_id_to_idx", {}):
                        duplicate_skipped += 1
                        continue
                    if tracker_i == "sam2":
                        mask_arg: Any = torch.from_numpy(mask.astype(np.float32))
                    else:
                        mask_arg = mask.astype(np.float32)
                    with torch.inference_mode(), _autocast_for_predictor(predictor, state["device"]):
                        predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=frame_idx_i,
                            obj_id=int(obj_id),
                            mask=mask_arg,
                        )
                    committed += 1
                    delays.append(max(0, int(current_frame_idx) - int(row.get("queued_at_frame_idx", frame_idx_i))))
        finally:
            state["tracking_has_started"] = old_tracking_started

        reconsolidated = False
        if old_tracking_started and int(len(state.get("obj_ids", []))) > old_obj_count:
            with torch.inference_mode(), _autocast_for_predictor(predictor, state["device"]):
                reconsolidate_timed(predictor, state)
            reconsolidated = True
        elapsed = float(time.time() - started)
        state["v107_birth_transaction_queue"] = []
        stats = _ROLLING_STATS
        trigger_key = {
            "min_pending": "birth_transaction_min_pending_trigger_count",
            "min_total_area": "birth_transaction_min_total_area_trigger_count",
            "max_delay": "birth_transaction_max_delay_trigger_count",
            "last_frame": "birth_transaction_last_frame_trigger_count",
            "high_value_immediate": "birth_transaction_immediate_trigger_count",
        }.get(str(reason))
        if trigger_key:
            stats[trigger_key] = int(stats[trigger_key]) + 1
        stats["birth_transaction_commit_count"] = int(stats["birth_transaction_commit_count"]) + 1
        stats["birth_transaction_committed_mask_count"] = int(stats["birth_transaction_committed_mask_count"]) + int(committed)
        stats["birth_transaction_commit_runtime_sec"] = float(stats["birth_transaction_commit_runtime_sec"]) + elapsed
        if reconsolidated:
            stats["birth_transaction_reconsolidate_call_count"] = int(
                stats["birth_transaction_reconsolidate_call_count"]
            ) + 1
        stats["birth_transaction_delay_frame_sum"] = int(stats["birth_transaction_delay_frame_sum"]) + int(sum(delays))
        if delays:
            stats["birth_transaction_max_delay_frames_observed"] = max(
                int(stats["birth_transaction_max_delay_frames_observed"]), int(max(delays))
            )
        _record_birth_transaction(
            {
                "event": "commit",
                "current_frame_idx": int(current_frame_idx),
                "reason": str(reason),
                "queued_entry_count": int(len(queue)),
                "committed_mask_count": int(committed),
                "duplicate_skipped_count": int(duplicate_skipped),
                "runtime_sec": elapsed,
                "reconsolidated": bool(reconsolidated),
                "old_object_count": int(old_obj_count),
                "new_object_count": int(len(state.get("obj_ids", []))),
                "delay_frames": [int(v) for v in delays],
            }
        )
        return int(committed)

    def add_masks_rolling(
        predictor: Any,
        state: dict[str, Any],
        *,
        tracker: str,
        frame_idx: int,
        obj_ids: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        if state.get("v106_rolling_state_enabled"):
            _rolling_add_frame(predictor, state, frame_idx=int(frame_idx), rgb=get_rgb(int(frame_idx)))
        input_count = int(masks.shape[0]) if masks.size else 0
        filtered_obj_ids, filtered_masks = _filter_post_start_births(
            state,
            frame_idx=int(frame_idx),
            obj_ids=obj_ids,
            masks=masks,
            rgb=get_rgb(int(frame_idx)) if bool(birth_appearance_enabled) else None,
            min_area=birth_min_area,
            max_area=birth_max_area,
            max_uncovered_ratio=birth_max_uncovered_ratio,
            max_bbox_frac=birth_max_bbox_frac,
            max_edge_touch_count=birth_max_edge_touch_count,
            min_extent=birth_min_extent,
            min_core_area=birth_min_core_area,
            shape_min_uncovered_ratio=birth_shape_min_uncovered_ratio,
            every=birth_every,
            max_per_frame=birth_max_per_frame,
            persistence_iou=birth_persistence_iou,
            persistence_hits=birth_persistence_hits,
            pending_ttl=birth_pending_ttl,
            persistence_min_area=birth_persistence_min_area,
            persistence_max_per_frame=birth_persistence_max_per_frame,
            immediate_area=birth_immediate_area,
            rescue_min_visible_count=birth_rescue_min_visible_count,
            rescue_min_foreground_ratio=birth_rescue_min_foreground_ratio,
            appearance_enabled=birth_appearance_enabled,
            appearance_min_iou=birth_appearance_min_iou,
            appearance_max_color_distance=birth_appearance_max_color_distance,
            appearance_max_centroid_distance=birth_appearance_max_centroid_distance,
            appearance_max_area_ratio=birth_appearance_max_area_ratio,
        )
        admitted_count = int(filtered_masks.shape[0]) if filtered_masks.size else 0
        stats = _ROLLING_STATS
        stats["stream_add_masks_call_count"] = int(stats["stream_add_masks_call_count"]) + 1
        stats["stream_add_masks_input_mask_count"] = int(stats["stream_add_masks_input_mask_count"]) + input_count
        stats["stream_add_masks_admitted_mask_count"] = int(stats["stream_add_masks_admitted_mask_count"]) + admitted_count
        stats["stream_add_masks_skipped_mask_count"] = int(stats["stream_add_masks_skipped_mask_count"]) + max(
            0, input_count - admitted_count
        )
        if admitted_count <= 0:
            return None
        existing = state.get("obj_id_to_idx", {})
        has_new_post_start_ids = bool(state.get("tracking_has_started", False)) and any(
            int(obj_id) not in existing for obj_id in filtered_obj_ids.tolist()
        )
        started = time.time()
        try:
            if bool(birth_transaction_enabled) and has_new_post_start_ids:
                areas = np.asarray([int(np.count_nonzero(mask)) for mask in filtered_masks.astype(bool)], dtype=np.int64)
                high_value_immediate = bool(
                    birth_transaction_immediate_area > 0
                    and int(np.count_nonzero(areas >= int(birth_transaction_immediate_area))) > 0
                )
                enqueue_birth_transaction(
                    state,
                    frame_idx=int(frame_idx),
                    tracker=str(tracker),
                    obj_ids=filtered_obj_ids,
                    masks=filtered_masks,
                )
                if high_value_immediate:
                    commit_birth_transaction(
                        predictor,
                        state,
                        current_frame_idx=int(frame_idx),
                        reason="high_value_immediate",
                    )
                    return None
                should_commit, reason = birth_transaction_trigger(state, current_frame_idx=int(frame_idx))
                if should_commit:
                    commit_birth_transaction(
                        predictor,
                        state,
                        current_frame_idx=int(frame_idx),
                        reason=str(reason),
                    )
                return None
            if state.get("v106_rolling_state_enabled") and birth_recon_prune_keep_frames > 0:
                _prune_stream_outputs_for_birth_recon(
                    state,
                    current_frame_idx=int(frame_idx),
                    keep_recent_frames=int(birth_recon_prune_keep_frames),
                )
                _prune_rolling_frame_store(state, current_frame_idx=int(frame_idx))
            return original_add_masks(
                predictor,
                state,
                tracker=tracker,
                frame_idx=int(frame_idx),
                obj_ids=filtered_obj_ids,
                masks=filtered_masks,
            )
        finally:
            stats["stream_add_masks_runtime_sec"] = float(stats["stream_add_masks_runtime_sec"]) + float(
                time.time() - started
            )

    def infer_rolling(predictor: Any, state: dict[str, Any], *, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if bool(birth_transaction_enabled):
            should_commit, reason = birth_transaction_trigger(state, current_frame_idx=int(frame_idx))
            if should_commit:
                commit_birth_transaction(
                    predictor,
                    state,
                    current_frame_idx=int(frame_idx),
                    reason=str(reason),
                )
        if state.get("v106_rolling_state_enabled"):
            _rolling_add_frame(predictor, state, frame_idx=int(frame_idx), rgb=get_rgb(int(frame_idx)))
        ids, masks = original_infer(predictor, state, frame_idx=int(frame_idx))
        visible_count = int(ids.shape[0]) if ids.size else 0
        foreground_ratio = 0.0
        if masks.size:
            union = np.any(masks.astype(bool, copy=False), axis=0)
            foreground_ratio = float(np.count_nonzero(union)) / float(union.size)
        state["v106_last_infer_frame_idx"] = int(frame_idx)
        state["v106_last_infer_visible_count"] = int(visible_count)
        state["v106_last_infer_foreground_ratio"] = float(foreground_ratio)
        return ids, masks

    def prune_stream_oversized_and_growth(
        predictor: Any,
        state: dict[str, Any] | None,
        *,
        visible_ids: np.ndarray,
        visible_masks: np.ndarray,
        max_visible_area: int,
        last_visible_frame_by_obj: dict[int, int],
        action: str = "prune",
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        kept_ids, kept_masks, events = original_prune_oversized(
            predictor,
            state,
            visible_ids=visible_ids,
            visible_masks=visible_masks,
            max_visible_area=max_visible_area,
            last_visible_frame_by_obj=last_visible_frame_by_obj,
            action=action,
        )
        reset_obj_ids: list[int] = []
        if state is not None:
            raw_reset_obj_ids = state.pop("v107_growth_history_reset_obj_ids", [])
            if isinstance(raw_reset_obj_ids, (list, tuple, set)):
                reset_obj_ids = [int(v) for v in raw_reset_obj_ids]
            elif raw_reset_obj_ids:
                reset_obj_ids = [int(raw_reset_obj_ids)]
        for reset_obj_id in reset_obj_ids:
            if reset_obj_id in growth_area_history_by_obj:
                growth_area_history_by_obj.pop(reset_obj_id, None)
                _ROLLING_STATS["stream_growth_prune_history_reset_count"] = int(
                    _ROLLING_STATS["stream_growth_prune_history_reset_count"]
                ) + 1
                _record_growth_prune(
                    {
                        "current_frame_idx": int(state.get("v106_current_frame_idx_for_birth_filter", -1))
                        if state is not None
                        else -1,
                        "history_reset_object_id": int(reset_obj_id),
                    }
                )
        if not growth_prune_enabled or kept_ids.size == 0 or kept_masks.size == 0:
            if kept_ids.size and kept_masks.size:
                for obj_id, mask in zip(kept_ids.tolist(), kept_masks.astype(bool), strict=False):
                    hist = growth_area_history_by_obj.setdefault(int(obj_id), [])
                    hist.append(int(np.count_nonzero(mask)))
                    del hist[:-growth_prune_history]
            return kept_ids, kept_masks, events

        areas = np.asarray([int(np.count_nonzero(mask)) for mask in kept_masks.astype(bool)], dtype=np.int64)
        growth_flags = np.zeros((int(kept_ids.shape[0]),), dtype=bool)
        growth_events: list[dict[str, Any]] = []
        for idx, (obj_id, area) in enumerate(zip(kept_ids.tolist(), areas.tolist(), strict=False)):
            obj_id_i = int(obj_id)
            area_i = int(area)
            hist = growth_area_history_by_obj.get(obj_id_i, [])
            if len(hist) < growth_prune_warmup or area_i < growth_prune_min_area:
                continue
            recent = np.asarray(hist[-growth_prune_history:], dtype=np.float64)
            baseline = float(np.median(recent)) if recent.size else 0.0
            ratio = float(area_i) / max(baseline, 1.0)
            if (
                growth_prune_max_history_median_area > 0
                and baseline > float(growth_prune_max_history_median_area)
            ):
                _ROLLING_STATS["stream_growth_prune_skipped_by_history_median_count"] = int(
                    _ROLLING_STATS["stream_growth_prune_skipped_by_history_median_count"]
                ) + 1
                continue
            if baseline > 0.0 and ratio >= growth_prune_ratio:
                growth_flags[int(idx)] = True
                growth_events.append(
                    {
                        "object_id": obj_id_i,
                        "visible_area": area_i,
                        "history_median_area": baseline,
                        "growth_ratio": ratio,
                        "growth_prune_ratio": float(growth_prune_ratio),
                        "growth_prune_min_area": int(growth_prune_min_area),
                        "growth_prune_history": int(growth_prune_history),
                        "growth_prune_warmup": int(growth_prune_warmup),
                        "growth_prune_max_history_median_area": int(
                            growth_prune_max_history_median_area
                        ),
                        "growth_prune": True,
                        "action": str(growth_prune_action),
                        "object_removed": False,
                        "output_suppressed": bool(growth_prune_action in {"prune", "suppress_output"}),
                    }
                )

        if bool(np.any(growth_flags)):
            state_ids = set(map(int, state.get("obj_ids", []))) if state is not None else set()
            for event in growth_events:
                obj_id_i = int(event["object_id"])
                removed = False
                if (
                    growth_prune_action == "prune"
                    and state is not None
                    and hasattr(predictor, "remove_object")
                    and obj_id_i in state_ids
                ):
                    if not base.object_removal_would_clear_conditioning(state, obj_id_i):
                        predictor.remove_object(state, obj_id_i, strict=False, need_output=False)
                        removed = True
                event["object_removed"] = bool(removed)
                if growth_prune_action == "prune":
                    last_visible_frame_by_obj.pop(obj_id_i, None)
                    growth_area_history_by_obj.pop(obj_id_i, None)
            if growth_prune_action == "prune":
                _ROLLING_STATS["stream_growth_prune_call_count"] = int(
                    _ROLLING_STATS["stream_growth_prune_call_count"]
                ) + 1
                _ROLLING_STATS["stream_growth_pruned_object_count"] = int(
                    _ROLLING_STATS["stream_growth_pruned_object_count"]
                ) + int(np.count_nonzero(growth_flags))
            _record_growth_prune(
                {
                    "current_frame_idx": int(state.get("v106_current_frame_idx_for_birth_filter", -1))
                    if state is not None
                    else -1,
                    "events": growth_events,
                    "action": str(growth_prune_action),
                }
            )

        final_keep = np.ones_like(growth_flags, dtype=bool) if growth_prune_action == "alert_only" else ~growth_flags
        final_ids = kept_ids[final_keep]
        final_masks = kept_masks[final_keep]
        for obj_id, area in zip(final_ids.tolist(), areas[final_keep].tolist(), strict=False):
            hist = growth_area_history_by_obj.setdefault(int(obj_id), [])
            hist.append(int(area))
            del hist[:-growth_prune_history]
        events.extend(growth_events)
        return final_ids, final_masks, events

    def prune_noncond_rolling(
        state: dict[str, Any],
        *,
        current_frame_idx: int,
        keep_recent_frames: int,
    ) -> list[int]:
        pruned = original_prune_noncond(
            state,
            current_frame_idx=int(current_frame_idx),
            keep_recent_frames=int(keep_recent_frames),
        )
        if state.get("v106_rolling_state_enabled"):
            _prune_rolling_frame_store(state, current_frame_idx=int(current_frame_idx))
        return pruned

    class _NoopAnnotatedImage:
        def save(self, *_args: Any, **_kwargs: Any) -> None:
            _ROLLING_STATS["visual_export_noop_save_count"] = int(
                _ROLLING_STATS["visual_export_noop_save_count"]
            ) + 1

    def overlay_label_noop(rgb: np.ndarray, _label: np.ndarray) -> np.ndarray:
        return rgb

    def annotate_frame_noop(*_args: Any, **_kwargs: Any) -> _NoopAnnotatedImage:
        return _NoopAnnotatedImage()

    def make_sheet_grid_noop(*_args: Any, **_kwargs: Any) -> None:
        _ROLLING_STATS["visual_export_noop_sheet_count"] = int(
            _ROLLING_STATS["visual_export_noop_sheet_count"]
        ) + 1

    def write_video_noop(*_args: Any, **_kwargs: Any) -> None:
        _ROLLING_STATS["visual_export_noop_video_count"] = int(
            _ROLLING_STATS["visual_export_noop_video_count"]
        ) + 1

    def imwrite_noop(*_args: Any, **_kwargs: Any) -> bool:
        _ROLLING_STATS["visual_export_noop_imwrite_count"] = int(
            _ROLLING_STATS["visual_export_noop_imwrite_count"]
        ) + 1
        return True

    region_mask_segment_call_count = 0

    def run_sam2_point_segment_choice_gap_filtered(*seg_args: Any, **seg_kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        nonlocal region_mask_segment_call_count
        masks, stats = original_run_sam2_point_segment_choice(*seg_args, **seg_kwargs)
        region_mask = seg_kwargs.get("region_mask")
        if region_mask is None and len(seg_args) >= 4:
            region_mask = seg_args[3]
        if region_mask is None:
            return masks, stats
        region_mask_segment_call_count += 1
        is_gap_call = region_mask_segment_call_count > 1
        if not is_gap_call:
            return masks, stats
        if disable_gap_birth:
            input_count = int(np.asarray(masks).shape[0]) if np.asarray(masks).ndim >= 3 else 0
            _ROLLING_STATS["gap_birth_disabled_count"] = int(_ROLLING_STATS["gap_birth_disabled_count"]) + 1
            _ROLLING_STATS["gap_birth_disabled_mask_count"] = int(
                _ROLLING_STATS["gap_birth_disabled_mask_count"]
            ) + int(input_count)
            empty_masks = np.zeros((0, *np.asarray(masks).shape[-2:]), dtype=bool)
            stats = dict(stats)
            stats["gap_birth_disabled"] = True
            stats["post_disjoint_mask_count_before_gap_birth_disable"] = int(input_count)
            stats["post_disjoint_mask_count"] = 0
            return empty_masks, stats
        region_bool = np.asarray(region_mask).astype(bool)
        current_uncovered_ratio = float(np.count_nonzero(region_bool)) / float(max(region_bool.size, 1))
        filtered_masks, filter_stats = _filter_masks_by_shape(
            masks,
            current_uncovered_ratio=current_uncovered_ratio,
            min_uncovered_ratio=gap_output_shape_min_uncovered_ratio,
            max_bbox_frac=gap_output_max_bbox_frac,
            max_edge_touch_count=gap_output_max_edge_touch_count,
            min_extent=gap_output_min_extent,
            min_core_area=gap_output_min_core_area,
            min_input_mask_count=gap_output_min_input_mask_count,
        )
        if bool(filter_stats.get("enabled", False)):
            _ROLLING_STATS["gap_output_filter_call_count"] = int(_ROLLING_STATS["gap_output_filter_call_count"]) + 1
            _ROLLING_STATS["gap_output_filter_input_mask_count"] = int(
                _ROLLING_STATS["gap_output_filter_input_mask_count"]
            ) + int(filter_stats.get("input_mask_count", 0))
            _ROLLING_STATS["gap_output_filter_kept_mask_count"] = int(
                _ROLLING_STATS["gap_output_filter_kept_mask_count"]
            ) + int(filter_stats.get("kept_mask_count", 0))
            _ROLLING_STATS["gap_output_filter_dropped_mask_count"] = int(
                _ROLLING_STATS["gap_output_filter_dropped_mask_count"]
            ) + int(filter_stats.get("dropped_mask_count", 0))
            if bool(filter_stats.get("active", False)):
                _ROLLING_STATS["gap_output_filter_active_call_count"] = int(
                    _ROLLING_STATS["gap_output_filter_active_call_count"]
                ) + 1
            _ROLLING_STATS["gap_output_filter_dropped_by_bbox_frac"] = int(
                _ROLLING_STATS["gap_output_filter_dropped_by_bbox_frac"]
            ) + len(filter_stats.get("dropped_by_bbox_frac_indices", []))
            _ROLLING_STATS["gap_output_filter_dropped_by_edge_touch"] = int(
                _ROLLING_STATS["gap_output_filter_dropped_by_edge_touch"]
            ) + len(filter_stats.get("dropped_by_edge_touch_indices", []))
            _ROLLING_STATS["gap_output_filter_dropped_by_extent"] = int(
                _ROLLING_STATS["gap_output_filter_dropped_by_extent"]
            ) + len(filter_stats.get("dropped_by_extent_indices", []))
            _ROLLING_STATS["gap_output_filter_dropped_by_core_area"] = int(
                _ROLLING_STATS["gap_output_filter_dropped_by_core_area"]
            ) + len(filter_stats.get("dropped_by_core_area_indices", []))
        stats = dict(stats)
        stats["gap_output_shape_filter"] = filter_stats
        stats["post_disjoint_mask_count_before_gap_output_filter"] = int(
            filter_stats.get("input_mask_count", int(np.asarray(masks).shape[0]))
        )
        stats["post_disjoint_mask_count"] = int(filtered_masks.shape[0])
        return filtered_masks, stats

    try:
        base.reconsolidate_stream_state_outputs = reconsolidate_timed
        base.setup_models = setup_models_rolling
        base.add_masks_to_stream_state = add_masks_rolling
        base.infer_stream_frame = infer_rolling
        base.prune_stream_oversized_visible_objects = prune_stream_oversized_and_growth
        base.prune_stream_noncond_memory = prune_noncond_rolling
        base.run_sam2_point_segment_choice = run_sam2_point_segment_choice_gap_filtered
        if skip_visual_export or lean_visual_export or label_only_visual_export:
            base.make_sheet_grid = make_sheet_grid_noop
            base.write_video = write_video_noop
        if skip_visual_export or label_only_visual_export:
            base.overlay_label = overlay_label_noop
            base.annotate_frame = annotate_frame_noop
        if skip_visual_export:
            if not str(getattr(args, "birth_dump_dir", "")).strip():
                base.cv2.imwrite = imwrite_noop
        base.run(args)
    finally:
        base.reconsolidate_stream_state_outputs = original_reconsolidate
        base.setup_models = original_setup_models
        base.add_masks_to_stream_state = original_add_masks
        base.infer_stream_frame = original_infer
        base.prune_stream_oversized_visible_objects = original_prune_oversized
        base.prune_stream_noncond_memory = original_prune_noncond
        base.run_sam2_point_segment_choice = original_run_sam2_point_segment_choice
        base.overlay_label = original_overlay_label
        base.annotate_frame = original_annotate_frame
        base.make_sheet_grid = original_make_sheet_grid
        base.write_video = original_write_video
        base.cv2.imwrite = original_imwrite
