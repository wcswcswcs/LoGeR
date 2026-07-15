#!/usr/bin/env python3
"""Phase 2 SAM2 feature-bank parity audit for Stream4D v105.

This helper validates the external frame-feature bank against real SAM2 APIs.
It is intentionally strict about what it proves: image prompt parity, video
feature-cache parity, and frame0-frozen tracking parity. It does not yet claim
full baseline-x gap-birth schedule parity unless the emitted gate says so.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAM3D_ROOT))

from sgq_v105.sam2_feature_bank import Sam2FrameFeatureBank


BASELINE_RUNNER = REPO_ROOT / "tools/audit_v105_baseline_x_sam2_twostage_tracking.py"


def _load_baseline_runner() -> Any:
    spec = importlib.util.spec_from_file_location("v105_baseline_x_runner_for_phase2", BASELINE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load baseline runner: {BASELINE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v105_baseline_x_runner_for_phase2"] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except Exception:
        return p.as_posix()


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = int(np.count_nonzero(aa | bb))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(aa & bb)) / float(union)


def _mask_rows(ref_masks: np.ndarray, bank_masks: np.ndarray, *, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ref_count = int(ref_masks.shape[0]) if ref_masks.ndim == 3 else 0
    bank_count = int(bank_masks.shape[0]) if bank_masks.ndim == 3 else 0
    count = min(ref_count, bank_count)
    for idx in range(count):
        rows.append(
            {
                "scope": prefix,
                "mask_index": int(idx),
                "ref_pixels": int(np.count_nonzero(ref_masks[idx])),
                "bank_pixels": int(np.count_nonzero(bank_masks[idx])),
                "iou": float(_mask_iou(ref_masks[idx], bank_masks[idx])),
            }
        )
    if ref_count != bank_count:
        rows.append(
            {
                "scope": prefix,
                "mask_index": -1,
                "ref_mask_count": int(ref_count),
                "bank_mask_count": int(bank_count),
                "iou": 0.0,
                "count_mismatch": True,
            }
        )
    return rows


def _tree_compare(ref: Any, bank: Any, *, path: str) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    if isinstance(ref, torch.Tensor) and isinstance(bank, torch.Tensor):
        if tuple(ref.shape) != tuple(bank.shape):
            rows.append(
                {
                    "path": path,
                    "shape_match": False,
                    "ref_shape": list(ref.shape),
                    "bank_shape": list(bank.shape),
                    "max_abs_diff": None,
                    "allclose": False,
                }
            )
            return rows
        diff = (ref.detach().float() - bank.detach().float()).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rows.append(
            {
                "path": path,
                "shape_match": True,
                "shape": list(ref.shape),
                "dtype_ref": str(ref.dtype),
                "dtype_bank": str(bank.dtype),
                "max_abs_diff": max_abs,
                "allclose": bool(torch.allclose(ref.detach().float(), bank.detach().float(), atol=1e-6, rtol=1e-5)),
            }
        )
        return rows
    if isinstance(ref, dict) and isinstance(bank, dict):
        for key in sorted(set(ref) | set(bank)):
            if key not in ref or key not in bank:
                rows.append({"path": f"{path}.{key}", "missing_key": True, "allclose": False})
            else:
                rows.extend(_tree_compare(ref[key], bank[key], path=f"{path}.{key}"))
        return rows
    if isinstance(ref, (list, tuple)) and isinstance(bank, (list, tuple)):
        if len(ref) != len(bank):
            rows.append({"path": path, "length_ref": len(ref), "length_bank": len(bank), "allclose": False})
        for idx, (a, b) in enumerate(zip(ref, bank, strict=False)):
            rows.extend(_tree_compare(a, b, path=f"{path}[{idx}]"))
        return rows
    rows.append({"path": path, "ref": str(ref), "bank": str(bank), "allclose": ref == bank})
    return rows


def _bound_point_segment_choice(
    mod: Any,
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
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torchvision.ops import nms

    try:
        from sam2.utils.amg import batched_mask_to_box, calculate_stability_score
    except Exception:
        from efficient_track_anything.utils.amg import batched_mask_to_box, calculate_stability_score

    h, w = rgb.shape[:2]
    if choice_policy not in {"largest_valid_mask_per_point", "smallest_valid_mask_per_point"}:
        raise ValueError(f"unsupported choice_policy={choice_policy}")
    if nms_score_type not in {"pred_iou", "stability"}:
        raise ValueError(f"unsupported nms_score_type={nms_score_type}")

    selected_batches = []
    selected_score_batches = []
    prompt_with_good = 0
    raw_option_count = 0
    batch_size = max(int(points_per_batch), 1)
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
            area_for_choice = areas.clone()
            if choice_policy == "largest_valid_mask_per_point":
                area_for_choice[~good] = -1
                chosen_idx = area_for_choice.argmax(dim=1)
            else:
                area_for_choice[~good] = torch.iinfo(torch.int64).max // 4
                chosen_idx = area_for_choice.argmin(dim=1)
            has_good = good.any(dim=1)
            prompt_with_good += int(has_good.sum().item())
            prompt_indices = torch.nonzero(has_good, as_tuple=False).flatten()
            if int(prompt_indices.numel()) > 0:
                selected = masks[prompt_indices, chosen_idx[prompt_indices]] > float(model_mask_thresh)
                if nms_score_type == "pred_iou":
                    selected_scores = iou_predictions[prompt_indices, chosen_idx[prompt_indices]].float()
                else:
                    selected_scores = stability[prompt_indices, chosen_idx[prompt_indices]].float()
                selected_batches.append(selected)
                selected_score_batches.append(selected_scores)

    if selected_batches:
        selected_t = torch.cat(selected_batches, dim=0)
        selected_scores_t = torch.cat(selected_score_batches, dim=0)
        if region_mask is not None:
            region_t = torch.as_tensor(region_mask.astype(bool), device="cuda")
            selected_t = selected_t & region_t.unsqueeze(0)
        pre_nms_count = int(selected_t.shape[0])
        if apply_box_nms and int(selected_t.shape[0]) > 0:
            boxes = batched_mask_to_box(selected_t).float()
            keep = nms(boxes, selected_scores_t, iou_threshold=float(box_nms_thresh))
            selected_t = selected_t[keep]
        selected_np = selected_t.detach().cpu().numpy().astype(bool)
    else:
        pre_nms_count = 0
        selected_np = np.zeros((0, h, w), dtype=bool)

    disjoint_np = mod.disjoin_smallest_first(
        selected_np,
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
        "pre_nms_mask_count": int(pre_nms_count),
        "post_disjoint_mask_count": int(disjoint_np.shape[0]),
        "apply_box_nms": bool(apply_box_nms),
        "nms_score_type": nms_score_type,
    }
    if apply_box_nms:
        stats["post_nms_mask_count"] = int(selected_np.shape[0])
    return disjoint_np, stats


def _run_image_prompt_parity(
    mod: Any,
    args: SimpleNamespace,
    segmentor: Any,
    bank: Sam2FrameFeatureBank,
    frame_ids: list[int],
    rgbs: list[np.ndarray],
) -> dict[str, Any]:
    h, w = rgbs[0].shape[:2]
    s1_seed = mod.stable_seed(args.seed, args.scene_id, frame_ids[0], args.stage1_num_pts, args.stage1_point_mode, "baseline-x-stage1-largest")
    s1_points, _ = mod.make_points_yx_torch(int(args.stage1_num_pts), s1_seed, str(args.stage1_point_mode))

    t0 = time.time()
    stage1_ref, stage1_ref_stats = mod.run_sam2_point_segment_choice(
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
    ref_s1_sec = time.time() - t0

    bank.bind_image_predictor(segmentor, frame_ids[0])
    t0 = time.time()
    stage1_bank, stage1_bank_stats = _bound_point_segment_choice(
        mod,
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
    bank_s1_sec = time.time() - t0

    uncovered0 = mod.uncovered_from_masks(stage1_ref, h, w)
    s2_seed = mod.stable_seed(args.seed, args.scene_id, frame_ids[0], args.stage2_num_pts, "baseline-x-stage2-smallest")
    s2_points = mod.sample_points_from_mask_yx(uncovered0, int(args.stage2_num_pts), s2_seed, inner_margin=int(args.gap_inner_margin))

    t0 = time.time()
    stage2_ref, stage2_ref_stats = mod.run_sam2_point_segment_choice(
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
    ref_s2_sec = time.time() - t0

    bank.bind_image_predictor(segmentor, frame_ids[0])
    t0 = time.time()
    stage2_bank, stage2_bank_stats = _bound_point_segment_choice(
        mod,
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
    bank_s2_sec = time.time() - t0

    rows = _mask_rows(stage1_ref, stage1_bank, prefix="frame0_stage1")
    rows.extend(_mask_rows(stage2_ref, stage2_bank, prefix="frame0_stage2_uncovered"))
    min_iou = min([float(row["iou"]) for row in rows], default=1.0)
    union_ref = np.any(np.concatenate([stage1_ref, stage2_ref], axis=0), axis=0) if stage1_ref.size or stage2_ref.size else np.zeros((h, w), dtype=bool)
    union_bank = np.any(np.concatenate([stage1_bank, stage2_bank], axis=0), axis=0) if stage1_bank.size or stage2_bank.size else np.zeros((h, w), dtype=bool)

    masks0 = np.concatenate([stage1_ref, stage2_ref], axis=0) if stage2_ref.size else stage1_ref
    return {
        "schema_version": "stream4d_v105_phase2_image_prompt_parity_v1",
        "stage1_ref_stats": stage1_ref_stats,
        "stage1_bank_stats": stage1_bank_stats,
        "stage2_ref_stats": stage2_ref_stats,
        "stage2_bank_stats": stage2_bank_stats,
        "stage1_ref_runtime_sec": float(ref_s1_sec),
        "stage1_bank_decoder_runtime_sec": float(bank_s1_sec),
        "stage2_ref_runtime_sec": float(ref_s2_sec),
        "stage2_bank_decoder_runtime_sec": float(bank_s2_sec),
        "per_mask_iou_rows": rows,
        "min_stage1_stage2_iou": float(min_iou),
        "foreground_union_iou": float(_mask_iou(union_ref, union_bank)),
        "stage1_count_match": int(stage1_ref.shape[0]) == int(stage1_bank.shape[0]),
        "stage2_count_match": int(stage2_ref.shape[0]) == int(stage2_bank.shape[0]),
        "parity_pass": bool(min_iou >= 0.999 and _mask_iou(union_ref, union_bank) >= 0.999),
        "frozen_frame0_masks": masks0,
    }


def _run_video_feature_parity(
    tracker_model: Any,
    bank: Sam2FrameFeatureBank,
    video_dir: Path,
    frame_ids: list[int],
    *,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
) -> dict[str, Any]:
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        direct_state = tracker_model.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=bool(offload_video_to_cpu),
            offload_state_to_cpu=bool(offload_state_to_cpu),
            async_loading_frames=False,
        )
        bank_state = tracker_model.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=bool(offload_video_to_cpu),
            offload_state_to_cpu=bool(offload_state_to_cpu),
            async_loading_frames=False,
        )
        bank.bind_video_state(bank_state)
        rows: list[dict[str, Any]] = []
        for chunk_index, frame_id in enumerate(frame_ids):
            direct_features = tracker_model._get_image_feature(direct_state, int(chunk_index), batch_size=1)
            bank_features = tracker_model._get_image_feature(bank_state, int(chunk_index), batch_size=1)
            cmp_rows = _tree_compare(direct_features, bank_features, path=f"frame[{chunk_index}]")
            for row in cmp_rows:
                row["frame_id"] = int(frame_id)
                row["chunk_index"] = int(chunk_index)
            rows.extend(cmp_rows)
    max_abs_diff = max([float(row.get("max_abs_diff") or 0.0) for row in rows], default=0.0)
    return {
        "schema_version": "stream4d_v105_phase2_video_feature_parity_v1",
        "row_count": int(len(rows)),
        "rows": rows,
        "max_abs_diff": float(max_abs_diff),
        "allclose": bool(all(bool(row.get("allclose", False)) for row in rows)),
    }


def _tracking_outputs(
    mod: Any,
    tracker_model: Any,
    video_dir: Path,
    *,
    tracker: str,
    masks0: np.ndarray,
    total_frames: int,
    offload_video_to_cpu: bool,
    offload_state_to_cpu: bool,
    bank: Sam2FrameFeatureBank | None,
) -> dict[int, dict[int, np.ndarray]]:
    import torch

    outputs: dict[int, dict[int, np.ndarray]] = {}
    obj_ids = np.arange(int(masks0.shape[0]), dtype=np.int64)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = tracker_model.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=bool(offload_video_to_cpu),
            offload_state_to_cpu=bool(offload_state_to_cpu),
            async_loading_frames=False,
        )
        if bank is not None:
            bank.bind_video_state(state)
        for obj_id, mask in zip(obj_ids.tolist(), masks0.astype(bool), strict=False):
            mask_arg: Any
            if tracker == "sam2":
                mask_arg = torch.from_numpy(mask.astype(np.float32))
            else:
                mask_arg = mask.astype(np.float32)
            out_frame_idx, out_obj_ids, out_mask_logits = tracker_model.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=int(obj_id),
                mask=mask_arg,
            )
            frame_outputs = outputs.setdefault(int(out_frame_idx), {})
            for idx, out_obj_id in enumerate(out_obj_ids):
                frame_outputs[int(out_obj_id)] = (
                    (out_mask_logits[idx] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                )
        for out_frame_idx, out_obj_ids, out_mask_logits in tracker_model.propagate_in_video(
            state,
            start_frame_idx=0,
            max_frame_num_to_track=int(total_frames),
        ):
            if int(out_frame_idx) >= int(total_frames):
                continue
            frame_outputs = outputs.setdefault(int(out_frame_idx), {})
            for idx, out_obj_id in enumerate(out_obj_ids):
                frame_outputs[int(out_obj_id)] = (
                    (out_mask_logits[idx] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                )
    return outputs


def _run_tracking_parity(
    mod: Any,
    args: SimpleNamespace,
    tracker_model: Any,
    bank: Sam2FrameFeatureBank,
    video_dir: Path,
    frame_ids: list[int],
    masks0: np.ndarray,
) -> dict[str, Any]:
    t0 = time.time()
    direct = _tracking_outputs(
        mod,
        tracker_model,
        video_dir,
        tracker=str(args.tracker_backend),
        masks0=masks0,
        total_frames=len(frame_ids),
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        offload_state_to_cpu=bool(args.offload_state_to_cpu),
        bank=None,
    )
    direct_sec = time.time() - t0
    t0 = time.time()
    bank_outputs = _tracking_outputs(
        mod,
        tracker_model,
        video_dir,
        tracker=str(args.tracker_backend),
        masks0=masks0,
        total_frames=len(frame_ids),
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        offload_state_to_cpu=bool(args.offload_state_to_cpu),
        bank=bank,
    )
    bank_sec = time.time() - t0

    rows: list[dict[str, Any]] = []
    for chunk_index, frame_id in enumerate(frame_ids):
        direct_frame = direct.get(int(chunk_index), {})
        bank_frame = bank_outputs.get(int(chunk_index), {})
        direct_ids = sorted(direct_frame)
        bank_ids = sorted(bank_frame)
        ids_match = direct_ids == bank_ids
        for obj_id in sorted(set(direct_ids) | set(bank_ids)):
            if obj_id not in direct_frame or obj_id not in bank_frame:
                rows.append(
                    {
                        "frame_id": int(frame_id),
                        "chunk_index": int(chunk_index),
                        "obj_id": int(obj_id),
                        "iou": 0.0,
                        "missing_direct": obj_id not in direct_frame,
                        "missing_bank": obj_id not in bank_frame,
                    }
                )
            else:
                rows.append(
                    {
                        "frame_id": int(frame_id),
                        "chunk_index": int(chunk_index),
                        "obj_id": int(obj_id),
                        "iou": float(_mask_iou(direct_frame[obj_id], bank_frame[obj_id])),
                    }
                )
        rows.append(
            {
                "frame_id": int(frame_id),
                "chunk_index": int(chunk_index),
                "obj_id": -1,
                "visible_id_set_match": bool(ids_match),
                "direct_visible_id_count": int(len(direct_ids)),
                "bank_visible_id_count": int(len(bank_ids)),
            }
        )
    iou_values = [float(row["iou"]) for row in rows if "iou" in row and int(row.get("obj_id", -1)) >= 0]
    visible_match = all(bool(row.get("visible_id_set_match", True)) for row in rows if int(row.get("obj_id", 0)) == -1)
    return {
        "schema_version": "stream4d_v105_phase2_frame0_frozen_tracking_parity_v1",
        "direct_tracking_runtime_sec": float(direct_sec),
        "bank_tracking_runtime_sec": float(bank_sec),
        "object_count": int(masks0.shape[0]),
        "row_count": int(len(rows)),
        "rows": rows,
        "min_tracking_mask_iou": float(min(iou_values, default=1.0)),
        "visible_id_sets_match": bool(visible_match),
        "parity_pass": bool(visible_match and min(iou_values, default=1.0) >= 0.995),
    }


def run_one_frame_count(
    *,
    mod: Any,
    args: SimpleNamespace,
    scene_id: str,
    frame_count: int,
    output_root: Path,
    baseline_config: Path,
) -> dict[str, Any]:
    import torch

    args.scene_id = scene_id
    args.frame_count = int(frame_count)
    args.frame_start = 0
    args.frame_stride = int(args.frame_stride)
    frame_ids = mod.parse_frame_ids("", int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = REPO_ROOT / str(args.rgb_root) / scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    rgbs = [_read_rgb(path) for path in frame_paths]

    case_dir = output_root / f"{scene_id}_f{int(frame_count):02d}"
    case_dir.mkdir(parents=True, exist_ok=True)
    models = mod.setup_models(args)
    segmentor = models["segmentor"]
    tracker_model = models["tracker_model"]

    image_bank = Sam2FrameFeatureBank(storage_device="cuda")
    image_bank.build_for_image_predictor(segmentor, frame_ids=frame_ids, rgb_frames=rgbs)
    image_parity = _run_image_prompt_parity(mod, args, segmentor, image_bank, frame_ids, rgbs)
    masks0 = image_parity.pop("frozen_frame0_masks")
    _write_json(case_dir / "image_feature_bank_summary.json", image_bank.summary())
    _write_json(case_dir / "image_prompt_parity_summary.json", image_parity)
    del image_bank
    torch.cuda.empty_cache()
    gc.collect()

    with tempfile.TemporaryDirectory(prefix="v105_phase2_video_") as tmp:
        video_dir = Path(tmp) / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        for chunk_idx, frame_path in enumerate(frame_paths):
            target = video_dir / f"{chunk_idx:05d}.jpg"
            target.write_bytes(frame_path.read_bytes())
        video_bank = Sam2FrameFeatureBank(storage_device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video_bank.build_for_video_paths(tracker_model, frame_ids=frame_ids, frame_paths=[video_dir / f"{idx:05d}.jpg" for idx in range(len(frame_ids))])
        video_feature_parity = _run_video_feature_parity(
            tracker_model,
            video_bank,
            video_dir,
            frame_ids,
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            offload_state_to_cpu=bool(args.offload_state_to_cpu),
        )
        tracking_parity = _run_tracking_parity(
            mod,
            args,
            tracker_model,
            video_bank,
            video_dir,
            frame_ids,
            masks0=masks0,
        )
        _write_json(case_dir / "video_feature_bank_summary.json", video_bank.summary())
        _write_json(case_dir / "video_feature_parity_summary.json", video_feature_parity)
        _write_json(case_dir / "frame0_frozen_tracking_parity_summary.json", tracking_parity)

    summary = {
        "schema_version": "stream4d_v105_phase2_feature_bank_case_summary_v1",
        "scene_id": scene_id,
        "frame_count": int(frame_count),
        "frame_ids": frame_ids,
        "baseline_config": _rel(baseline_config),
        "image_prompt_parity_pass": bool(image_parity["parity_pass"]),
        "video_feature_parity_pass": bool(video_feature_parity["allclose"]),
        "frame0_frozen_tracking_parity_pass": bool(tracking_parity["parity_pass"]),
        "min_stage1_stage2_iou": float(image_parity["min_stage1_stage2_iou"]),
        "foreground_union_iou": float(image_parity["foreground_union_iou"]),
        "video_feature_max_abs_diff": float(video_feature_parity["max_abs_diff"]),
        "min_tracking_mask_iou": float(tracking_parity["min_tracking_mask_iou"]),
        "visible_id_sets_match": bool(tracking_parity["visible_id_sets_match"]),
        "status": "pass"
        if image_parity["parity_pass"] and video_feature_parity["allclose"] and tracking_parity["parity_pass"]
        else "fail",
        "scope_note": (
            "This validates image prompt parity, video feature-cache parity, and frame0-frozen tracking parity. "
            "It is not yet full baseline-x gap-birth schedule parity."
        ),
        "artifacts": {
            "image_feature_bank_summary": _rel(case_dir / "image_feature_bank_summary.json"),
            "image_prompt_parity_summary": _rel(case_dir / "image_prompt_parity_summary.json"),
            "video_feature_bank_summary": _rel(case_dir / "video_feature_bank_summary.json"),
            "video_feature_parity_summary": _rel(case_dir / "video_feature_parity_summary.json"),
            "frame0_frozen_tracking_parity_summary": _rel(case_dir / "frame0_frozen_tracking_parity_summary.json"),
        },
    }
    _write_json(case_dir / "case_summary.json", summary)
    del models
    torch.cuda.empty_cache()
    gc.collect()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--frame-counts", default="2,8,32")
    parser.add_argument("--baseline-config", default="configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml")
    parser.add_argument("--output-root", default="Stream3D/outputs/audit/v105_specgap_phase2_feature_bank_parity_20260711")
    parser.add_argument("--frame-stride", type=int, default=5)
    args_cli = parser.parse_args()

    mod = _load_baseline_runner()
    baseline_config = REPO_ROOT / args_cli.baseline_config
    cfg = mod.load_config(baseline_config)
    args = mod.make_args(
        cfg,
        argparse.Namespace(
            config=baseline_config,
            scene_id=args_cli.scene_id,
            rgb_root=None,
            frame_start=0,
            frame_stride=int(args_cli.frame_stride),
            frame_count=2,
            frame_ids="",
            output_root=str(REPO_ROOT / "Stream3D/outputs/audit/v105_phase2_unused_baseline_runner_output"),
            seed=None,
        ),
    )
    frame_counts = [int(part.strip()) for part in str(args_cli.frame_counts).split(",") if part.strip()]
    output_root = Path(args_cli.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    case_summaries = []
    for frame_count in frame_counts:
        case_summaries.append(
            run_one_frame_count(
                mod=mod,
                args=args,
                scene_id=args_cli.scene_id,
                frame_count=int(frame_count),
                output_root=output_root,
                baseline_config=baseline_config,
            )
        )
    aggregate = {
        "schema_version": "stream4d_v105_phase2_feature_bank_parity_aggregate_v1",
        "scene_id": args_cli.scene_id,
        "frame_counts": frame_counts,
        "case_summaries": case_summaries,
        "all_cases_pass": bool(all(row["status"] == "pass" for row in case_summaries)),
        "scope_note": (
            "Aggregate covers image prompt parity, video feature-cache parity, and frame0-frozen tracking parity. "
            "Full baseline-x frozen gap-birth schedule parity remains a separate gate."
        ),
    }
    _write_json(output_root / "aggregate_summary.json", aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
