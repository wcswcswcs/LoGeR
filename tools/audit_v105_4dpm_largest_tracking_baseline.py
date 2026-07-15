#!/usr/bin/env python3
"""4D_PM-style largest-candidate tracking ablation for v105.

This runner keeps the 4D_PM video structure: initialize masks with SAM2 point
prompts, propagate new masks with a video tracker, and fill uncovered regions
with SAM2 gap prompts. The ablation changes only the per-point multimask choice:
instead of the smallest valid SAM2 mask, choose the largest valid mask.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_RGB_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"
DEFAULT_OUT = STREAM3D_ROOT / "outputs" / "audit" / "v105_4dpm_largest_sam2seg_tracking_scene0011_r1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    annotate_frame,
    disjoin_smallest_first,
    make_points_yx_torch,
    mask_stats,
    overlay_label,
    parse_frame_ids,
    read_rgb,
    sha256_file,
    stable_seed,
)


def clear_sam2_namespace(repo_path: Path) -> None:
    for name in list(sys.modules):
        if name == "sam2" or name.startswith("sam2."):
            del sys.modules[name]
    repo = str(repo_path)
    sys.path[:] = [p for p in sys.path if p != repo]
    sys.path.insert(0, repo)


def make_numeric_frame_dir(frame_paths: list[Path], output_dir: Path) -> Path:
    frame_dir = output_dir / "_numeric_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(frame_paths):
        dst = frame_dir / f"{idx:06d}.jpg"
        os.symlink(os.path.abspath(src), dst)
    return frame_dir


def clone_empty_video_state_template(template_state: dict[str, Any]) -> dict[str, Any]:
    ordered_dict_type = type(template_state.get("obj_id_to_idx", {}))
    return {
        "images": template_state["images"],
        "num_frames": int(template_state["num_frames"]),
        "offload_video_to_cpu": bool(template_state["offload_video_to_cpu"]),
        "offload_state_to_cpu": bool(template_state["offload_state_to_cpu"]),
        "video_height": template_state["video_height"],
        "video_width": template_state["video_width"],
        "device": template_state["device"],
        "storage_device": template_state["storage_device"],
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "cached_features": {},
        "constants": {},
        "obj_id_to_idx": ordered_dict_type(),
        "obj_idx_to_id": ordered_dict_type(),
        "obj_ids": [],
        "output_dict": {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        },
        "output_dict_per_obj": {},
        "temp_output_dict_per_obj": {},
        "consolidated_frame_inds": {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        },
        "tracking_has_started": False,
        "frames_already_tracked": {},
    }


def make_sheet_grid(frame_paths: list[Path], out_path: Path, cell_width: int, cols: int = 4) -> None:
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    if not images:
        raise RuntimeError("no frames for sheet")
    resized = []
    for image in images:
        ratio = float(cell_width) / float(max(image.width, 1))
        resized.append(image.resize((int(cell_width), int(round(image.height * ratio))), Image.Resampling.LANCZOS))
    cell_h = max(img.height for img in resized)
    rows = int(np.ceil(len(resized) / float(cols)))
    canvas = Image.new("RGB", (cols * int(cell_width), rows * cell_h), (0, 0, 0))
    for idx, image in enumerate(resized):
        rr, cc = divmod(idx, cols)
        canvas.paste(image, (cc * int(cell_width), rr * cell_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def write_video(frame_paths: list[Path], out_path: Path, fps: float) -> None:
    if not frame_paths:
        raise RuntimeError("no frames for video")
    first = np.asarray(Image.open(frame_paths[0]).convert("RGB"))
    h, w = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out_path}")
    try:
        for path in frame_paths:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            if rgb.shape[:2] != (h, w):
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def sample_points_from_mask_yx(mask_np: np.ndarray, n: int, seed: int, *, inner_margin: int = 2) -> Any:
    import torch
    import torch.nn.functional as F

    mask = torch.as_tensor(mask_np.astype(bool), device="cuda")
    h, w = mask.shape
    weights = mask.float().unsqueeze(0).unsqueeze(0)
    if inner_margin > 0:
        r = int(inner_margin)
        weights = 1.0 - F.max_pool2d(1.0 - weights, kernel_size=2 * r + 1, stride=1, padding=r)
        if float(weights.sum().item()) <= 0.0:
            weights = mask.float().unsqueeze(0).unsqueeze(0)
    flat = weights.reshape(-1)
    if float(flat.sum().item()) <= 0.0:
        return torch.zeros((0, 2), device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    idx = torch.multinomial(flat, int(n), replacement=True, generator=generator)
    ys = (idx // int(w)).float()
    xs = (idx % int(w)).float()
    y_norm = 2.0 * ys / float(max(h - 1, 1)) - 1.0
    x_norm = 2.0 * xs / float(max(w - 1, 1)) - 1.0
    return torch.stack([y_norm, x_norm], dim=1)


def disjoin_keep_order(masks: np.ndarray, h: int, w: int, empty_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if masks.size == 0:
        return np.zeros((0, h, w), dtype=bool), np.zeros((0,), dtype=bool)
    if masks.ndim == 2:
        masks = masks[None, :, :]
    claimed = np.zeros((h, w), dtype=bool)
    out = []
    keep = []
    min_pixels = int(h * w * float(empty_ratio))
    for mask in masks.astype(bool):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        residual = mask & ~claimed
        kept = int(np.count_nonzero(residual)) > min_pixels
        out.append(residual)
        keep.append(kept)
        claimed |= mask
    return np.stack(out, axis=0).astype(bool), np.asarray(keep, dtype=bool)


def label_from_id_masks(obj_ids: np.ndarray, masks: np.ndarray, h: int, w: int) -> np.ndarray:
    label = np.zeros((h, w), dtype=np.uint16)
    if masks.size == 0:
        return label
    for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        label[mask] = int(obj_id) + 1
    return label


def setup_models(args: argparse.Namespace) -> dict[str, Any]:
    tracker = str(args.tracker)
    if tracker == "sam2":
        repo_path = REPO_ROOT / "Grounded-SAM-2"
        clear_sam2_namespace(repo_path)
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam2_checkpoint = REPO_ROOT / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        image_model = build_sam2(sam2_cfg, str(sam2_checkpoint), device="cuda")
        segmentor = SAM2ImagePredictor(image_model)
        tracker_model = build_sam2_video_predictor(sam2_cfg, str(sam2_checkpoint), device="cuda")
        tracker_checkpoint = sam2_checkpoint
        tracker_cfg = sam2_cfg
    elif tracker == "edgetam":
        repo_path = REPO_ROOT / "third_party" / "EdgeTAM"
        clear_sam2_namespace(repo_path)
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        initialize_config_module(config_module="sam2", version_base="1.3.2")
        sam2_checkpoint = REPO_ROOT / "Grounded-SAM-2" / "checkpoints" / "sam2.1_hiera_large.pt"
        sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        image_model = build_sam2(sam2_cfg, str(sam2_checkpoint), device="cuda")
        segmentor = SAM2ImagePredictor(image_model)
        tracker_checkpoint = REPO_ROOT / "third_party" / "EdgeTAM" / "checkpoints" / "edgetam.pt"
        tracker_cfg = "configs/edgetam.yaml"
        tracker_model = build_sam2_video_predictor(tracker_cfg, str(tracker_checkpoint), device="cuda")
    else:
        raise ValueError(tracker)

    return {
        "segmentor": segmentor,
        "tracker_model": tracker_model,
        "sam2_checkpoint": sam2_checkpoint,
        "sam2_cfg": sam2_cfg,
        "tracker_checkpoint": tracker_checkpoint,
        "tracker_cfg": tracker_cfg,
    }


def run_sam2_point_segment(
    segmentor: Any,
    rgb: np.ndarray,
    *,
    points_yx: Any,
    region_mask: np.ndarray | None,
    points_per_batch: int,
    iou_threshold: float,
    stability_threshold: float,
    stability_score_offset: float,
    model_mask_thresh: float,
    box_nms_thresh: float,
    empty_ratio: float,
    apply_box_nms: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from sam2.utils.amg import batched_mask_to_box, calculate_stability_score
    from torchvision.ops import nms

    h, w = rgb.shape[:2]
    segmentor.reset_predictor()
    segmentor.set_image(rgb)
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
            area_for_choice[~good] = -1
            has_good = good.any(dim=1)
            prompt_with_good += int(has_good.sum().item())
            chosen_idx = area_for_choice.argmax(dim=1)
            prompt_indices = torch.nonzero(has_good, as_tuple=False).flatten()
            if int(prompt_indices.numel()) > 0:
                selected = masks[prompt_indices, chosen_idx[prompt_indices]] > float(model_mask_thresh)
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

    disjoint_np = disjoin_smallest_first(
        selected_np,
        h,
        w,
        empty_ratio=float(empty_ratio),
        fix_small_regions=True,
    )
    stats = {
        "raw_multimask_option_count": int(raw_option_count),
        "prompt_with_good_mask_count": int(prompt_with_good),
        "pre_nms_mask_count": int(pre_nms_count),
        "post_disjoint_mask_count": int(disjoint_np.shape[0]),
    }
    if apply_box_nms:
        stats["post_nms_mask_count"] = int(selected_np.shape[0])
    return disjoint_np, stats


def propagate_new_masks(
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
    feature_bank_frame_offset: int = 0,
    state_num_frames_override: int | None = None,
    clear_cached_features_after_init: bool = False,
    video_state_template: dict[str, Any] | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    import torch

    outputs: dict[int, dict[int, np.ndarray]] = {}
    if masks.size == 0:
        return outputs
    if video_state_template is None:
        state = predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=bool(offload_video_to_cpu),
            offload_state_to_cpu=bool(offload_state_to_cpu),
            async_loading_frames=False,
        )
    else:
        state = clone_empty_video_state_template(video_state_template)
    state["_v105_feature_bank_frame_offset"] = int(feature_bank_frame_offset)
    if clear_cached_features_after_init:
        state["cached_features"] = {}
    if state_num_frames_override is not None:
        state["num_frames"] = int(state_num_frames_override)
    try:
        for obj_id, mask in zip(obj_ids.tolist(), masks.astype(bool), strict=False):
            mask_arg: Any
            if tracker == "sam2":
                mask_arg = torch.from_numpy(mask.astype(np.float32))
            else:
                mask_arg = mask.astype(np.float32)
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=state,
                frame_idx=int(seed_frame),
                obj_id=int(obj_id),
                mask=mask_arg,
            )
            frame_outputs = outputs.setdefault(int(out_frame_idx), {})
            for idx, out_obj_id in enumerate(out_obj_ids):
                frame_outputs[int(out_obj_id)] = (
                    (out_mask_logits[idx] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                )
        max_frames = max(0, int(total_frames) - int(seed_frame))
        if max_frames > 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=int(seed_frame),
                max_frame_num_to_track=int(max_frames),
            ):
                if int(out_frame_idx) >= int(total_frames):
                    continue
                frame_outputs = outputs.setdefault(int(out_frame_idx), {})
                for idx, obj_id in enumerate(out_obj_ids):
                    frame_outputs[int(obj_id)] = (
                        (out_mask_logits[idx] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                    )
    finally:
        try:
            predictor.reset_state(state)
        except Exception:
            pass
        try:
            state.clear()
        except Exception:
            pass
    return outputs


def run(args: argparse.Namespace) -> None:
    import torch

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = Path(args.rgb_root).resolve() / args.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])

    variant = f"sam2seg_largest_{args.tracker}_tracker"
    output_root = Path(args.output_root).resolve() / variant
    label_dir = output_root / "labels"
    overlay_dir = output_root / "overlays"
    sheet_dir = output_root / "sheets"
    for directory in (label_dir, overlay_dir, sheet_dir, output_root / "videos"):
        directory.mkdir(parents=True, exist_ok=True)
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
    init_seed = stable_seed(args.seed, args.scene_id, frame_ids[0], args.num_pts, args.point_mode, "init-largest")
    init_points, point_meta = make_points_yx_torch(int(args.num_pts), init_seed, str(args.point_mode))
    t0 = time.time()
    masks0, init_stats = run_sam2_point_segment(
        segmentor,
        rgbs[0],
        points_yx=init_points,
        region_mask=None,
        points_per_batch=int(args.points_per_batch),
        iou_threshold=float(args.iou_threshold),
        stability_threshold=float(args.stability_threshold),
        stability_score_offset=float(args.stability_score_offset),
        model_mask_thresh=float(args.model_mask_thresh),
        box_nms_thresh=float(args.box_nms_thresh),
        empty_ratio=float(args.empty_ratio),
        apply_box_nms=True,
    )
    init_runtime_sec = time.time() - t0
    next_obj_id = int(masks0.shape[0])
    per_frame_masks[0] = masks0
    per_frame_ids[0] = np.arange(next_obj_id, dtype=np.int64)
    to_prop_masks[0] = masks0
    to_prop_ids[0] = per_frame_ids[0]
    frame_diagnostics.append(
        {
            "chunk_frame_index": 0,
            "frame_id": int(frame_ids[0]),
            "init_mask_count": int(masks0.shape[0]),
            "init_runtime_sec": float(init_runtime_sec),
            **init_stats,
        }
    )

    total_tracking_sec = 0.0
    total_gap_seg_sec = 0.0
    empty_propagation_frames = 0
    for t in range(len(frame_ids) - 1):
        prop_t0 = time.time()
        propagated = propagate_new_masks(
            tracker_model,
            tracker=str(args.tracker),
            video_dir=video_dir,
            seed_frame=t,
            obj_ids=to_prop_ids[t],
            masks=to_prop_masks[t],
            total_frames=len(frame_ids),
            offload_video_to_cpu=bool(args.offload_video_to_cpu),
            offload_state_to_cpu=bool(args.offload_state_to_cpu),
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
            current_masks_all, keep = disjoin_keep_order(
                current_masks_pre,
                h,
                w,
                empty_ratio=float(args.empty_ratio),
            )
            current_masks = current_masks_all[keep]
            current_ids = current_ids[keep]
        else:
            empty_propagation_frames += 1
            current_ids = np.zeros((0,), dtype=np.int64)
            current_masks = np.zeros((0, h, w), dtype=bool)
            current_masks_pre = np.zeros((0, h, w), dtype=bool)

        if current_masks.size:
            uncovered = ~np.any(current_masks, axis=0)
        else:
            uncovered = np.ones((h, w), dtype=bool)
        gap_seed = stable_seed(args.seed, args.scene_id, frame_ids[t + 1], args.num_pts_active, args.point_mode, "gap-largest")
        gap_points = sample_points_from_mask_yx(
            uncovered,
            int(args.num_pts_active),
            gap_seed,
            inner_margin=int(args.gap_inner_margin),
        )
        gap_t0 = time.time()
        if int(gap_points.shape[0]) > 0:
            gap_masks, gap_stats = run_sam2_point_segment(
                segmentor,
                rgbs[t + 1],
                points_yx=gap_points,
                region_mask=uncovered,
                points_per_batch=int(args.points_per_batch),
                iou_threshold=float(args.active_iou_threshold),
                stability_threshold=float(args.active_stability_threshold),
                stability_score_offset=float(args.stability_score_offset),
                model_mask_thresh=float(args.model_mask_thresh),
                box_nms_thresh=float(args.box_nms_thresh),
                empty_ratio=float(args.empty_ratio),
                apply_box_nms=False,
            )
        else:
            gap_masks = np.zeros((0, h, w), dtype=bool)
            gap_stats = {
                "raw_multimask_option_count": 0,
                "prompt_with_good_mask_count": 0,
                "pre_nms_mask_count": 0,
                "post_disjoint_mask_count": 0,
            }
        gap_sec = time.time() - gap_t0
        total_gap_seg_sec += gap_sec
        gap_ids = np.arange(next_obj_id, next_obj_id + int(gap_masks.shape[0]), dtype=np.int64)
        next_obj_id += int(gap_masks.shape[0])
        to_prop_masks[t + 1] = gap_masks
        to_prop_ids[t + 1] = gap_ids
        if gap_masks.size:
            per_frame_masks[t + 1] = np.concatenate([current_masks, gap_masks], axis=0)
            per_frame_ids[t + 1] = np.concatenate([current_ids, gap_ids], axis=0)
        else:
            per_frame_masks[t + 1] = current_masks
            per_frame_ids[t + 1] = current_ids
        frame_diagnostics.append(
            {
                "chunk_frame_index": int(t + 1),
                "frame_id": int(frame_ids[t + 1]),
                "propagation_seed_frame_index": int(t),
                "propagation_runtime_sec": float(prop_sec),
                "new_seed_mask_count": int(to_prop_masks[t].shape[0]),
                "propagated_pre_disjoin_count": int(current_masks_pre.shape[0]),
                "propagated_post_disjoin_count": int(current_masks.shape[0]),
                "gap_runtime_sec": float(gap_sec),
                "gap_mask_count": int(gap_masks.shape[0]),
                "final_frame_mask_count": int(per_frame_masks[t + 1].shape[0]),
                "uncovered_ratio_before_gap": float(np.count_nonzero(uncovered)) / float(uncovered.size),
                "gap_stats": gap_stats,
            }
        )
        print(
            json.dumps(
                {
                    "tracker": str(args.tracker),
                    "frame_index": int(t + 1),
                    "frame_id": int(frame_ids[t + 1]),
                    "propagated": int(current_masks.shape[0]),
                    "gap": int(gap_masks.shape[0]),
                    "final": int(per_frame_masks[t + 1].shape[0]),
                    "prop_sec": round(prop_sec, 3),
                    "gap_sec": round(gap_sec, 3),
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
                f"select=largest points={int(args.num_pts)} active={int(args.num_pts_active)}",
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
    summary = {
        "schema_version": "stream4d_v105_4dpm_largest_tracking_summary_v1",
        "variant": variant,
        "scene_id": str(args.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "frame_count": int(len(frame_ids)),
        "segmentor": "sam2.1_hiera_large",
        "tracker": str(args.tracker),
        "selection_policy": "largest_valid_mask_per_point",
        "point_sampling": point_meta,
        "num_pts": int(args.num_pts),
        "num_pts_active": int(args.num_pts_active),
        "points_per_batch": int(args.points_per_batch),
        "iou_threshold": float(args.iou_threshold),
        "stability_threshold": float(args.stability_threshold),
        "active_iou_threshold": float(args.active_iou_threshold),
        "active_stability_threshold": float(args.active_stability_threshold),
        "box_nms_thresh": float(args.box_nms_thresh),
        "empty_ratio": float(args.empty_ratio),
        "setup_sec": float(setup_sec),
        "total_runtime_sec": float(total_sec),
        "initial_segmentation_runtime_sec": float(init_runtime_sec),
        "total_tracking_runtime_sec": float(total_tracking_sec),
        "total_gap_segmentation_runtime_sec": float(total_gap_seg_sec),
        "empty_propagation_frames": int(empty_propagation_frames),
        "initial_mask_count": int(masks0.shape[0]),
        "total_object_id_count": int(next_obj_id),
        "mean_visible_id_count": float(np.mean([row["visible_id_count"] for row in records])) if records else 0.0,
        "mean_foreground_ratio": float(np.mean([row["foreground_ratio"] for row in records])) if records else 0.0,
        "peak_cuda_memory_mb": float(peak_mb),
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
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "video": str(video_path), "sheets": sheet_paths}, ensure_ascii=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, choices=["sam2", "edgetam"])
    parser.add_argument("--scene-id", default="scene0011_00")
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=32)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT))
    parser.add_argument("--num-pts", type=int, default=800)
    parser.add_argument("--num-pts-active", type=int, default=800)
    parser.add_argument("--point-mode", default="random_seeded", choices=["random_seeded", "4dpm_grid"])
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--points-per-batch", type=int, default=800)
    parser.add_argument("--iou-threshold", type=float, default=0.8)
    parser.add_argument("--stability-threshold", type=float, default=0.8)
    parser.add_argument("--active-iou-threshold", type=float, default=0.8)
    parser.add_argument("--active-stability-threshold", type=float, default=0.8)
    parser.add_argument("--stability-score-offset", type=float, default=1.0)
    parser.add_argument("--model-mask-thresh", type=float, default=0.0)
    parser.add_argument("--box-nms-thresh", type=float, default=0.8)
    parser.add_argument("--empty-ratio", type=float, default=0.001)
    parser.add_argument("--gap-inner-margin", type=int, default=2)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sheet-cell-width", type=int, default=520)
    parser.add_argument("--fps", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
