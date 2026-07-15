#!/usr/bin/env python3
"""Build a non-oracle residual gap-birth bank for v106 handoff repair.

The tool only uses RGB frames, inherited v106 handoff birth records, and the
labels produced by replaying those inherited records. It deliberately does not
read the independent reference labels or reference birth masks.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"
DEFAULT_RGB_ROOT = STREAM3D_ROOT / "data" / "scannet" / "processed"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from Stream3D.stream4d_v106.artifacts import sha256_file, write_json  # noqa: E402
from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    disjoin_smallest_first,
    dump_birth_masks,
    load_config,
    make_args,
    run_sam2_point_segment_choice,
    sample_component_adaptive_points_yx,
    setup_models,
    uncovered_from_masks,
)
from tools.audit_v105_4dpm_largest_tracking_baseline import sample_points_from_mask_yx  # noqa: E402
from tools.audit_v105_4dpm_style_per_frame_segmentors import make_points_yx_torch, read_rgb, stable_seed  # noqa: E402


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _rel(path_text: str | Path) -> str:
    path = _resolve(path_text)
    try:
        return str(path.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_label(path: Path, h: int, w: int) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[:2] != (h, w):
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.uint16)


def _label_to_masks(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = sorted(int(v) for v in np.unique(label) if int(v) > 0)
    if not values:
        h, w = label.shape[:2]
        return np.zeros((0,), dtype=np.int64), np.zeros((0, h, w), dtype=bool)
    obj_ids = np.asarray([int(v) - 1 for v in values], dtype=np.int64)
    masks = np.stack([(label == int(v)) for v in values], axis=0).astype(bool)
    return obj_ids, masks


def _make_baseline_args(config_path: Path, cli: argparse.Namespace, frame_ids: list[int]) -> SimpleNamespace:
    config = load_config(config_path)
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=cli.scene_id,
        rgb_root=str(cli.rgb_root),
        frame_start=int(frame_ids[0]) if frame_ids else 0,
        frame_stride=0,
        frame_count=len(frame_ids),
        frame_ids=",".join(str(v) for v in frame_ids),
        output_root=str(cli.output_root),
        seed=cli.seed,
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(cli.output_root)
    if cli.model_dtype:
        args.model_dtype = str(cli.model_dtype)
    return args


def _override_int(value: int | None, fallback: int) -> int:
    return int(fallback if value is None else value)


def _override_float(value: float | None, fallback: float) -> float:
    return float(fallback if value is None else value)


def _birth_admission_enabled(cli: argparse.Namespace) -> bool:
    return (
        int(cli.max_birth_mask_area) > 0
        or float(cli.max_birth_mask_area_ratio) > 0.0
        or float(cli.max_birth_mask_uncovered_ratio) > 0.0
    )


def _apply_birth_admission_policy(
    masks: np.ndarray,
    *,
    uncovered: np.ndarray,
    cli: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Apply non-oracle mask sanity gates before new global IDs are allocated."""
    h, w = uncovered.shape[:2]
    image_area = int(h * w)
    uncovered_area = int(np.count_nonzero(uncovered))
    policy = {
        "enabled": bool(_birth_admission_enabled(cli)),
        "max_birth_mask_area": int(cli.max_birth_mask_area),
        "max_birth_mask_area_ratio": float(cli.max_birth_mask_area_ratio),
        "max_birth_mask_uncovered_ratio": float(cli.max_birth_mask_uncovered_ratio),
        "image_area": int(image_area),
        "uncovered_area": int(uncovered_area),
    }
    if masks.size == 0:
        summary = dict(policy)
        summary.update(
            {
                "pre_admission_mask_count": 0,
                "post_admission_mask_count": 0,
                "filtered_birth_mask_count": 0,
                "filtered_by_reason": {},
                "max_mask_area": 0,
                "max_mask_area_ratio": 0.0,
                "max_mask_uncovered_ratio": 0.0,
            }
        )
        return np.zeros((0, h, w), dtype=bool), [], summary

    kept_masks: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    filtered_by_reason: dict[str, int] = {}
    max_area = 0
    max_area_ratio = 0.0
    max_uncovered_ratio = 0.0
    for raw_index, mask in enumerate(masks.astype(bool)):
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(np.count_nonzero(mask))
        area_ratio = float(area / max(image_area, 1))
        uncovered_overlap = int(np.count_nonzero(mask & uncovered.astype(bool)))
        uncovered_ratio = float(area / max(uncovered_area, 1))
        max_area = max(max_area, area)
        max_area_ratio = max(max_area_ratio, area_ratio)
        max_uncovered_ratio = max(max_uncovered_ratio, uncovered_ratio)
        reject_reasons: list[str] = []
        if int(cli.max_birth_mask_area) > 0 and area > int(cli.max_birth_mask_area):
            reject_reasons.append("area_gt_max_birth_mask_area")
        if float(cli.max_birth_mask_area_ratio) > 0.0 and area_ratio > float(cli.max_birth_mask_area_ratio):
            reject_reasons.append("area_ratio_gt_max_birth_mask_area_ratio")
        if float(cli.max_birth_mask_uncovered_ratio) > 0.0 and uncovered_ratio > float(cli.max_birth_mask_uncovered_ratio):
            reject_reasons.append("uncovered_ratio_gt_max_birth_mask_uncovered_ratio")
        accepted = not reject_reasons
        meta = {
            "raw_mask_index": int(raw_index),
            "accepted": bool(accepted),
            "mask_area": int(area),
            "mask_area_ratio": float(area_ratio),
            "mask_uncovered_overlap_pixels": int(uncovered_overlap),
            "mask_uncovered_ratio": float(uncovered_ratio),
            "reject_reasons": reject_reasons,
        }
        if accepted:
            kept_masks.append(mask.astype(bool))
            metas.append(meta)
        else:
            for reason in reject_reasons:
                filtered_by_reason[reason] = int(filtered_by_reason.get(reason, 0)) + 1

    if kept_masks:
        accepted_masks = np.stack(kept_masks, axis=0).astype(bool)
    else:
        accepted_masks = np.zeros((0, h, w), dtype=bool)
    summary = dict(policy)
    summary.update(
        {
            "pre_admission_mask_count": int(masks.shape[0]),
            "post_admission_mask_count": int(accepted_masks.shape[0]),
            "filtered_birth_mask_count": int(masks.shape[0] - accepted_masks.shape[0]),
            "filtered_by_reason": filtered_by_reason,
            "max_mask_area": int(max_area),
            "max_mask_area_ratio": float(max_area_ratio),
            "max_mask_uncovered_ratio": float(max_uncovered_ratio),
        }
    )
    return accepted_masks, metas, summary


def _best_label_overlap(mask: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    mask_bool = mask.astype(bool)
    mask_area = int(np.count_nonzero(mask_bool))
    best = {
        "best_overlap_obj_id": None,
        "best_overlap_label_value": None,
        "best_overlap_coeff": 0.0,
        "best_intersection_pixels": 0,
        "best_label_area": 0,
        "candidate_area": int(mask_area),
    }
    if mask_area <= 0:
        return best
    for label_value in sorted(int(v) for v in np.unique(label) if int(v) > 0):
        label_mask = label == int(label_value)
        label_area = int(np.count_nonzero(label_mask))
        inter = int(np.count_nonzero(mask_bool & label_mask))
        coeff = float(inter / max(1, min(mask_area, label_area)))
        if coeff > float(best["best_overlap_coeff"]):
            best = {
                "best_overlap_obj_id": int(label_value) - 1,
                "best_overlap_label_value": int(label_value),
                "best_overlap_coeff": float(coeff),
                "best_intersection_pixels": int(inter),
                "best_label_area": int(label_area),
                "candidate_area": int(mask_area),
            }
    return best


def _classify_repair_birth_defer_candidate(
    *,
    mask: np.ndarray,
    label: np.ndarray,
    cli: argparse.Namespace,
) -> dict[str, Any]:
    mode = str(cli.repair_birth_defer_mode)
    overlap = _best_label_overlap(mask, label)
    area = int(overlap["candidate_area"])
    if mode == "off":
        return {
            "mode": mode,
            "action": "birth_new",
            "target_obj_id": None,
            "reason": "repair_birth_defer_disabled",
            **overlap,
        }
    if mode != "overlap":
        raise ValueError(f"unsupported repair_birth_defer_mode={mode}")
    if area < int(cli.repair_birth_defer_min_area):
        return {
            "mode": mode,
            "action": "noise",
            "target_obj_id": None,
            "reason": "candidate_area_below_repair_birth_defer_min_area",
            **overlap,
        }
    coeff = float(overlap["best_overlap_coeff"])
    target_obj_id = overlap.get("best_overlap_obj_id")
    if target_obj_id is not None and coeff >= float(cli.duplicate_suppress_overlap_coeff):
        return {
            "mode": mode,
            "action": "duplicate_suppress_existing",
            "target_obj_id": int(target_obj_id),
            "reason": "overlap_coeff_above_duplicate_suppress_threshold",
            **overlap,
        }
    if target_obj_id is not None and coeff >= float(cli.repair_overlap_coeff):
        return {
            "mode": mode,
            "action": "repair_existing",
            "target_obj_id": int(target_obj_id),
            "reason": "overlap_coeff_above_repair_threshold",
            **overlap,
        }
    if coeff <= float(cli.birth_max_overlap_coeff):
        return {
            "mode": mode,
            "action": "birth_new",
            "target_obj_id": None,
            "reason": "overlap_coeff_below_birth_threshold",
            **overlap,
        }
    ambiguous_action = str(cli.ambiguous_overlap_action)
    return {
        "mode": mode,
        "action": ambiguous_action,
        "target_obj_id": None,
        "reason": "ambiguous_overlap_between_birth_and_repair_thresholds",
        **overlap,
    }


def _temporal_residual_repair_enabled(cli: argparse.Namespace) -> bool:
    return str(cli.temporal_residual_repair_mode) != "off"


def _best_temporal_residual_match(
    *,
    mask: np.ndarray,
    registry: list[dict[str, Any]],
    chunk_frame_index: int,
    cli: argparse.Namespace,
) -> dict[str, Any] | None:
    """Find a non-oracle previous residual birth whose mask persists over time."""
    if not _temporal_residual_repair_enabled(cli):
        return None
    mode = str(cli.temporal_residual_repair_mode)
    if mode != "mask_overlap":
        raise ValueError(f"unsupported temporal_residual_repair_mode={mode}")
    candidate = mask.astype(bool)
    candidate_area = int(np.count_nonzero(candidate))
    min_area = int(cli.temporal_residual_min_area)
    if candidate_area < min_area:
        return None
    best: dict[str, Any] | None = None
    max_area_ratio = float(cli.temporal_residual_max_area_ratio)
    window = int(cli.temporal_residual_window_chunks)
    for item in registry:
        item_chunk = int(item["chunk_frame_index"])
        if window > 0 and int(chunk_frame_index) - item_chunk > window:
            continue
        prior = item["mask"].astype(bool)
        prior_area = int(item["area"])
        if prior_area < min_area:
            continue
        inter = int(np.count_nonzero(candidate & prior))
        if inter <= 0:
            continue
        union = int(candidate_area + prior_area - inter)
        min_overlap = float(inter / max(1, min(candidate_area, prior_area)))
        iou = float(inter / max(1, union))
        area_ratio = float(max(candidate_area, prior_area) / max(1, min(candidate_area, prior_area)))
        if min_overlap < float(cli.temporal_residual_min_overlap):
            continue
        if max_area_ratio > 0.0 and area_ratio > max_area_ratio:
            continue
        record = {
            "target_obj_id": int(item["obj_id"]),
            "target_chunk_frame_index": int(item_chunk),
            "target_frame_id": int(item["frame_id"]),
            "candidate_area": int(candidate_area),
            "target_area": int(prior_area),
            "intersection_pixels": int(inter),
            "union_pixels": int(union),
            "min_overlap": float(min_overlap),
            "iou": float(iou),
            "area_ratio": float(area_ratio),
            "source": str(item.get("source", "")),
        }
        key = (float(record["min_overlap"]), float(record["iou"]), int(record["intersection_pixels"]))
        if best is None:
            best = record
        else:
            best_key = (float(best["min_overlap"]), float(best["iou"]), int(best["intersection_pixels"]))
            if key > best_key:
                best = record
    return best


def _stack_bool_masks(masks: list[np.ndarray], h: int, w: int) -> np.ndarray:
    if not masks:
        return np.zeros((0, h, w), dtype=bool)
    return np.stack([mask.astype(bool) for mask in masks], axis=0).astype(bool)


def _largest_component_admission(
    mask: np.ndarray,
    *,
    min_area: int,
    min_largest_ratio: float,
    keep_largest_only: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return an admitted mask after optional largest-component stabilization."""
    bool_mask = mask.astype(bool)
    input_area = int(np.count_nonzero(bool_mask))
    stats: dict[str, Any] = {
        "input_area": int(input_area),
        "component_count": 0,
        "largest_component_area": 0,
        "largest_component_ratio": 0.0,
        "min_area": int(min_area),
        "min_largest_component_ratio": float(min_largest_ratio),
        "keep_largest_only": bool(keep_largest_only),
        "accepted": False,
        "reason": "empty_mask",
    }
    if input_area <= 0:
        return None, stats
    num_labels, labels, component_stats, _centroids = cv2.connectedComponentsWithStats(bool_mask.astype(np.uint8), 8)
    component_count = max(0, int(num_labels) - 1)
    stats["component_count"] = int(component_count)
    if component_count <= 0:
        stats["reason"] = "no_foreground_components"
        return None, stats
    areas = component_stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest_offset = int(np.argmax(areas))
    largest_label = int(largest_offset + 1)
    largest_area = int(areas[largest_offset])
    largest_ratio = float(largest_area / max(input_area, 1))
    stats["largest_component_area"] = int(largest_area)
    stats["largest_component_ratio"] = float(largest_ratio)
    if largest_area < int(min_area):
        stats["reason"] = "largest_component_below_min_area"
        return None, stats
    if largest_ratio < float(min_largest_ratio):
        stats["reason"] = "largest_component_ratio_below_threshold"
        return None, stats
    admitted = (labels == largest_label) if keep_largest_only else bool_mask
    admitted_area = int(np.count_nonzero(admitted))
    stats["accepted"] = True
    stats["reason"] = "accepted"
    stats["output_area"] = int(admitted_area)
    stats["removed_area"] = int(input_area - admitted_area)
    return admitted.astype(bool), stats


def _negative_points_for_positive(
    *,
    parent_mask: np.ndarray,
    positive_yx: tuple[int, int],
    count: int,
    min_distance_px: int,
) -> list[tuple[int, int]]:
    if count <= 0:
        return []
    ys, xs = np.nonzero(parent_mask.astype(bool))
    if len(ys) == 0:
        return []
    py, px = int(positive_yx[0]), int(positive_yx[1])
    dist2 = (ys.astype(np.int64) - py) ** 2 + (xs.astype(np.int64) - px) ** 2
    min_dist2 = int(max(min_distance_px, 0)) ** 2
    order = np.argsort(-dist2, kind="stable")
    chosen: list[tuple[int, int]] = []
    for idx in order:
        if int(dist2[int(idx)]) < min_dist2:
            break
        y, x = int(ys[int(idx)]), int(xs[int(idx)])
        if all((y - cy) ** 2 + (x - cx) ** 2 >= min_dist2 for cy, cx in chosen):
            chosen.append((y, x))
            if len(chosen) >= int(count):
                break
    return chosen


def _run_sam2_parent_conditioned_child_choice(
    segmentor: Any,
    rgb: np.ndarray,
    *,
    points_yx: Any,
    parent_mask: np.ndarray,
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
    negative_points_per_positive: int,
    negative_min_distance_px: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """SAM2 point decoding with optional parent-internal negative prompts."""
    if int(negative_points_per_positive) <= 0:
        masks, stats = run_sam2_point_segment_choice(
            segmentor,
            rgb,
            points_yx=points_yx,
            region_mask=parent_mask,
            points_per_batch=int(points_per_batch),
            choice_policy=str(choice_policy),
            iou_threshold=float(iou_threshold),
            stability_threshold=float(stability_threshold),
            stability_score_offset=float(stability_score_offset),
            model_mask_thresh=float(model_mask_thresh),
            box_nms_thresh=float(box_nms_thresh),
            empty_ratio=float(empty_ratio),
            apply_box_nms=bool(apply_box_nms),
            nms_score_type=str(nms_score_type),
        )
        stats["negative_points_per_positive"] = 0
        stats["negative_min_distance_px"] = int(negative_min_distance_px)
        return masks, stats

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

    points_np = points_yx.detach().cpu().numpy() if hasattr(points_yx, "detach") else np.asarray(points_yx)
    prompt_coords_yx: list[list[tuple[float, float]]] = []
    prompt_labels: list[list[int]] = []
    negative_count_total = 0
    for point in points_np:
        y_norm = float(point[0])
        x_norm = float(point[1])
        py = int(round((y_norm + 1.0) * 0.5 * float(max(h - 1, 1))))
        px = int(round((x_norm + 1.0) * 0.5 * float(max(w - 1, 1))))
        negatives = _negative_points_for_positive(
            parent_mask=parent_mask,
            positive_yx=(py, px),
            count=int(negative_points_per_positive),
            min_distance_px=int(negative_min_distance_px),
        )
        coords = [(y_norm, x_norm)]
        labels = [1]
        for ny, nx in negatives:
            coords.append(
                (
                    2.0 * float(ny) / float(max(h - 1, 1)) - 1.0,
                    2.0 * float(nx) / float(max(w - 1, 1)) - 1.0,
                )
            )
            labels.append(0)
        negative_count_total += len(negatives)
        prompt_coords_yx.append(coords)
        prompt_labels.append(labels)

    max_prompt_len = max([len(v) for v in prompt_coords_yx], default=0)
    if max_prompt_len <= 0:
        return np.zeros((0, h, w), dtype=bool), {
            "choice_policy": choice_policy,
            "raw_multimask_option_count": 0,
            "prompt_with_good_mask_count": 0,
            "pre_nms_mask_count": 0,
            "post_disjoint_mask_count": 0,
            "apply_box_nms": bool(apply_box_nms),
            "nms_score_type": str(nms_score_type),
            "negative_points_per_positive": int(negative_points_per_positive),
            "negative_min_distance_px": int(negative_min_distance_px),
            "negative_point_count": 0,
        }
    for coords, labels in zip(prompt_coords_yx, prompt_labels, strict=True):
        while len(coords) < max_prompt_len:
            coords.append(coords[0])
            labels.append(-1)

    coords_yx = torch.tensor(prompt_coords_yx, device="cuda", dtype=torch.float32)
    labels_t = torch.tensor(prompt_labels, device="cuda", dtype=torch.int)
    segmentor.reset_predictor()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentor.set_image(rgb)

    selected_batches = []
    selected_score_batches = []
    prompt_with_good = 0
    raw_option_count = 0
    batch_size = max(int(points_per_batch), 1)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, int(coords_yx.shape[0]), batch_size):
            coords_batch_yx = coords_yx[start : start + batch_size]
            labels_batch = labels_t[start : start + batch_size]
            pts_px = 0.5 * torch.tensor([h - 1, w - 1], device="cuda", dtype=torch.float32) * (
                coords_batch_yx + 1.0
            )
            pts_px = pts_px.round().long().flip(-1).float()
            coords = segmentor._transforms.transform_coords(pts_px, normalize=True, orig_hw=(h, w))
            masks, iou_predictions, _ = segmentor._predict(
                coords,
                labels_batch,
                multimask_output=True,
                return_logits=True,
            )
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
                if str(nms_score_type) == "pred_iou":
                    selected_scores = iou_predictions[prompt_indices, chosen_idx[prompt_indices]].float()
                else:
                    selected_scores = stability[prompt_indices, chosen_idx[prompt_indices]].float()
                selected_batches.append(selected)
                selected_score_batches.append(selected_scores)

    if selected_batches:
        selected_t = torch.cat(selected_batches, dim=0)
        selected_scores_t = torch.cat(selected_score_batches, dim=0)
        region_t = torch.as_tensor(parent_mask.astype(bool), device="cuda")
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
        "choice_policy": choice_policy,
        "iou_threshold": float(iou_threshold),
        "stability_threshold": float(stability_threshold),
        "raw_multimask_option_count": int(raw_option_count),
        "prompt_with_good_mask_count": int(prompt_with_good),
        "pre_nms_mask_count": int(pre_nms_count),
        "post_disjoint_mask_count": int(disjoint_np.shape[0]),
        "apply_box_nms": bool(apply_box_nms),
        "nms_score_type": str(nms_score_type),
        "negative_points_per_positive": int(negative_points_per_positive),
        "negative_min_distance_px": int(negative_min_distance_px),
        "negative_point_count": int(negative_count_total),
    }
    if apply_box_nms:
        stats["post_nms_mask_count"] = int(selected_np.shape[0])
    return disjoint_np, stats


def _apply_frame0_child_split(
    *,
    segmentor: Any,
    rgb: np.ndarray,
    masks: np.ndarray,
    source_records: list[dict[str, Any]],
    args: SimpleNamespace,
    cli: argparse.Namespace,
    frame_id: int,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Optionally split high-risk frame0 objectness masks using non-oracle child prompts."""
    h, w = rgb.shape[:2]
    mode = str(cli.frame0_child_split_mode)
    policy = {
        "enabled": bool(mode != "none"),
        "mode": mode,
        "target_source": str(cli.frame0_child_target_source),
        "min_parent_area_ratio": float(cli.frame0_child_min_parent_area_ratio),
        "max_parent_area_ratio": float(cli.frame0_child_max_parent_area_ratio),
        "max_parent_count": int(cli.frame0_child_max_parent_count),
        "parent_raw_index_allowlist": (
            [int(v) for v in sorted(cli.frame0_child_parent_raw_index_set)]
            if cli.frame0_child_parent_raw_index_set is not None
            else []
        ),
        "points_per_parent": int(cli.frame0_child_points_per_parent),
        "max_children_per_parent": int(cli.frame0_child_max_children_per_parent),
        "negative_points_per_positive": int(cli.frame0_child_negative_points_per_positive),
        "negative_min_distance_px": int(cli.frame0_child_negative_min_distance_px),
        "min_component_area": int(cli.frame0_child_min_component_area),
        "largest_component_only": bool(cli.frame0_child_largest_component_only),
        "min_largest_component_ratio": float(cli.frame0_child_min_largest_component_ratio),
        "parent_residual_min_area": int(cli.frame0_child_parent_residual_min_area),
        "choice_policy": str(cli.frame0_child_choice_policy),
    }
    if mode == "none" or masks.size == 0:
        summary = dict(policy)
        summary.update(
            {
                "input_mask_count": int(masks.shape[0]) if masks.ndim == 3 else 0,
                "output_mask_count": int(masks.shape[0]) if masks.ndim == 3 else 0,
                "target_parent_count": 0,
                "child_mask_count": 0,
                "parent_residual_count": 0,
                "runtime_sec": 0.0,
                "parent_records": [],
            }
        )
        return masks.astype(bool), source_records, summary
    if len(source_records) != int(masks.shape[0]):
        raise ValueError(f"source_records length {len(source_records)} does not match masks {masks.shape[0]}")
    if mode not in {"append", "replace_parent", "parent_minus_children"}:
        raise ValueError(f"unsupported frame0_child_split_mode={mode}")

    def is_target(raw_index: int, record: dict[str, Any], area_ratio: float) -> bool:
        parent_index_set = getattr(cli, "frame0_child_parent_raw_index_set", None)
        if parent_index_set is not None and int(raw_index) not in set(parent_index_set):
            return False
        target_source = str(cli.frame0_child_target_source)
        if target_source == "stage2_only" and str(record.get("stage")) != "stage2":
            return False
        if target_source == "stage1_only" and str(record.get("stage")) != "stage1":
            return False
        return (
            float(area_ratio) >= float(cli.frame0_child_min_parent_area_ratio)
            and float(area_ratio) <= float(cli.frame0_child_max_parent_area_ratio)
        )

    t0 = time.time()
    final_masks: list[np.ndarray] = []
    final_sources: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    target_parent_count = 0
    child_mask_count = 0
    parent_residual_count = 0
    image_area = max(1, int(h * w))
    max_parent_count = max(0, int(cli.frame0_child_max_parent_count))

    for raw_index, (mask, record) in enumerate(zip(masks.astype(bool), source_records, strict=True)):
        area = int(np.count_nonzero(mask))
        area_ratio = float(area / image_area)
        targeted = is_target(raw_index, record, area_ratio) and target_parent_count < max_parent_count
        if not targeted:
            copied = dict(record)
            copied["frame0_child_split_role"] = "untouched"
            final_masks.append(mask.astype(bool))
            final_sources.append(copied)
            continue

        target_parent_count += 1
        seed = stable_seed(
            int(cli.seed),
            str(cli.scene_id),
            int(frame_id),
            int(raw_index),
            int(cli.frame0_child_points_per_parent),
            "v106-frame0-child-split",
        )
        child_points, child_point_meta = sample_component_adaptive_points_yx(
            mask,
            max_points=int(cli.frame0_child_points_per_parent),
            min_component_area=int(cli.frame0_child_min_component_area),
            base_points_per_component=int(cli.frame0_child_points_per_parent),
            area_per_extra_point=max(image_area, 1),
            max_points_per_component=int(cli.frame0_child_points_per_parent),
            seed=seed,
        )
        child_t0 = time.time()
        if int(child_points.shape[0]) > 0:
            child_masks, child_stats = _run_sam2_parent_conditioned_child_choice(
                segmentor,
                rgb,
                points_yx=child_points,
                parent_mask=mask,
                points_per_batch=int(args.points_per_batch),
                choice_policy=str(cli.frame0_child_choice_policy),
                iou_threshold=float(cli.frame0_child_iou_threshold),
                stability_threshold=float(cli.frame0_child_stability_threshold),
                stability_score_offset=float(args.stability_score_offset),
                model_mask_thresh=float(args.model_mask_thresh),
                box_nms_thresh=float(args.box_nms_thresh),
                empty_ratio=float(args.empty_ratio),
                apply_box_nms=bool(cli.frame0_child_apply_box_nms),
                nms_score_type=str(cli.frame0_child_nms_score_type),
                negative_points_per_positive=int(cli.frame0_child_negative_points_per_positive),
                negative_min_distance_px=int(cli.frame0_child_negative_min_distance_px),
            )
        else:
            child_masks = np.zeros((0, h, w), dtype=bool)
            child_stats = {
                "choice_policy": str(cli.frame0_child_choice_policy),
                "raw_multimask_option_count": 0,
                "prompt_with_good_mask_count": 0,
                "pre_nms_mask_count": 0,
                "post_disjoint_mask_count": 0,
                "apply_box_nms": bool(cli.frame0_child_apply_box_nms),
                "nms_score_type": str(cli.frame0_child_nms_score_type),
                "negative_points_per_positive": int(cli.frame0_child_negative_points_per_positive),
                "negative_min_distance_px": int(cli.frame0_child_negative_min_distance_px),
                "negative_point_count": 0,
            }
        child_sec = time.time() - child_t0
        raw_child_count = int(child_masks.shape[0])
        max_children_per_parent = int(cli.frame0_child_max_children_per_parent)
        child_keep_indices = list(range(raw_child_count))
        if max_children_per_parent > 0 and raw_child_count > max_children_per_parent:
            child_areas = [int(np.count_nonzero(child_masks[idx].astype(bool))) for idx in child_keep_indices]
            ranked = sorted(
                zip(child_keep_indices, child_areas, strict=True),
                key=lambda item: (-int(item[1]), int(item[0])),
            )
            child_keep_indices = [idx for idx, _area in ranked[:max_children_per_parent]]
            child_keep_indices.sort()
            child_masks = child_masks[child_keep_indices].astype(bool)
            child_stats["pre_child_admission_mask_count"] = int(raw_child_count)
            child_stats["post_child_admission_mask_count"] = int(child_masks.shape[0])
            child_stats["child_admission_policy"] = {
                "max_children_per_parent": int(max_children_per_parent),
                "rank_key": "area_desc_then_original_index",
            }
        else:
            child_stats["pre_child_admission_mask_count"] = int(raw_child_count)
            child_stats["post_child_admission_mask_count"] = int(raw_child_count)
            child_stats["child_admission_policy"] = {
                "max_children_per_parent": int(max_children_per_parent),
                "rank_key": "area_desc_then_original_index",
            }
        component_admission_enabled = bool(cli.frame0_child_largest_component_only) or (
            float(cli.frame0_child_min_largest_component_ratio) > 0.0
        )
        if component_admission_enabled:
            pre_component_count = int(child_masks.shape[0])
            admitted_masks: list[np.ndarray] = []
            admitted_indices: list[int] = []
            component_records: list[dict[str, Any]] = []
            for component_candidate_idx, child_mask in enumerate(child_masks.astype(bool)):
                original_child_idx = (
                    int(child_keep_indices[component_candidate_idx])
                    if component_candidate_idx < len(child_keep_indices)
                    else int(component_candidate_idx)
                )
                admitted_mask, component_record = _largest_component_admission(
                    child_mask,
                    min_area=int(cli.frame0_child_min_component_area),
                    min_largest_ratio=float(cli.frame0_child_min_largest_component_ratio),
                    keep_largest_only=bool(cli.frame0_child_largest_component_only),
                )
                component_record.update(
                    {
                        "component_candidate_index": int(component_candidate_idx),
                        "original_child_index": int(original_child_idx),
                    }
                )
                component_records.append(component_record)
                if admitted_mask is not None:
                    admitted_masks.append(admitted_mask.astype(bool))
                    admitted_indices.append(int(original_child_idx))
            child_masks = _stack_bool_masks(admitted_masks, h, w)
            child_keep_indices = admitted_indices
            child_stats["pre_component_admission_mask_count"] = int(pre_component_count)
            child_stats["post_component_admission_mask_count"] = int(child_masks.shape[0])
            child_stats["component_admission_policy"] = {
                "largest_component_only": bool(cli.frame0_child_largest_component_only),
                "min_largest_component_ratio": float(cli.frame0_child_min_largest_component_ratio),
                "min_largest_component_area": int(cli.frame0_child_min_component_area),
            }
            child_stats["component_admission_records"] = component_records
        child_count = int(child_masks.shape[0])
        child_mask_count += child_count
        child_union = np.any(child_masks.astype(bool), axis=0) if child_count else np.zeros((h, w), dtype=bool)
        residual = mask.astype(bool) & ~child_union
        residual_area = int(np.count_nonzero(residual))

        parent_record = {
            "parent_raw_mask_index": int(raw_index),
            "parent_stage": str(record.get("stage")),
            "parent_stage_mask_index": record.get("stage_mask_index"),
            "parent_area": int(area),
            "parent_area_ratio": float(area_ratio),
            "child_point_count": int(child_points.shape[0]),
            "raw_child_mask_count": int(raw_child_count),
            "child_mask_count": int(child_count),
            "child_keep_indices": [int(idx) for idx in child_keep_indices],
            "child_union_area": int(np.count_nonzero(child_union)),
            "parent_residual_area": int(residual_area),
            "child_sampling": child_point_meta,
            "child_stats": child_stats,
            "child_runtime_sec": float(child_sec),
        }
        parent_records.append(parent_record)

        child_sources = []
        for emitted_child_idx, child_mask in enumerate(child_masks.astype(bool)):
            original_child_idx = int(child_keep_indices[emitted_child_idx]) if emitted_child_idx < len(child_keep_indices) else emitted_child_idx
            child_source = dict(record)
            child_source.update(
                {
                    "frame0_child_split_role": "child",
                    "frame0_child_parent_raw_mask_index": int(raw_index),
                    "frame0_child_parent_stage": str(record.get("stage")),
                    "frame0_child_parent_area_ratio": float(area_ratio),
                    "frame0_child_index": int(original_child_idx),
                    "frame0_child_emitted_index": int(emitted_child_idx),
                }
            )
            child_sources.append(child_source)

        parent_residual_source = dict(record)
        parent_residual_source.update(
            {
                "frame0_child_split_role": "parent_residual",
                "frame0_child_parent_raw_mask_index": int(raw_index),
                "frame0_child_parent_stage": str(record.get("stage")),
                "frame0_child_parent_area_ratio": float(area_ratio),
                "frame0_child_count": int(child_count),
            }
        )
        parent_original_source = dict(record)
        parent_original_source.update(
            {
                "frame0_child_split_role": "parent_original",
                "frame0_child_parent_raw_mask_index": int(raw_index),
                "frame0_child_parent_stage": str(record.get("stage")),
                "frame0_child_parent_area_ratio": float(area_ratio),
                "frame0_child_count": int(child_count),
            }
        )

        if mode == "append":
            final_masks.append(mask.astype(bool))
            final_sources.append(parent_original_source)
            for child_mask, child_source in zip(child_masks.astype(bool), child_sources, strict=False):
                final_masks.append(child_mask.astype(bool))
                final_sources.append(child_source)
        elif mode == "replace_parent":
            if child_count:
                for child_mask, child_source in zip(child_masks.astype(bool), child_sources, strict=False):
                    final_masks.append(child_mask.astype(bool))
                    final_sources.append(child_source)
            else:
                parent_original_source["frame0_child_split_role"] = "parent_original_no_child_fallback"
                final_masks.append(mask.astype(bool))
                final_sources.append(parent_original_source)
        else:
            if child_count:
                for child_mask, child_source in zip(child_masks.astype(bool), child_sources, strict=False):
                    final_masks.append(child_mask.astype(bool))
                    final_sources.append(child_source)
                if residual_area >= int(cli.frame0_child_parent_residual_min_area):
                    final_masks.append(residual.astype(bool))
                    final_sources.append(parent_residual_source)
                    parent_residual_count += 1
            else:
                parent_original_source["frame0_child_split_role"] = "parent_original_no_child_fallback"
                final_masks.append(mask.astype(bool))
                final_sources.append(parent_original_source)

    output_masks = _stack_bool_masks(final_masks, h, w)
    summary = dict(policy)
    summary.update(
        {
            "input_mask_count": int(masks.shape[0]),
            "output_mask_count": int(output_masks.shape[0]),
            "target_parent_count": int(target_parent_count),
            "child_mask_count": int(child_mask_count),
            "parent_residual_count": int(parent_residual_count),
            "runtime_sec": float(time.time() - t0),
            "parent_records": parent_records,
        }
    )
    return output_masks, final_sources, summary


def _selected_frame(chunk_idx: int, uncovered_ratio: float, cli: argparse.Namespace, emitted_count: int) -> tuple[bool, str]:
    selected_chunk_indices = getattr(cli, "selected_chunk_index_set", None)
    if selected_chunk_indices is not None and int(chunk_idx) not in selected_chunk_indices:
        return False, "not_in_selected_chunk_indices"
    if int(cli.start_chunk_index) > int(chunk_idx):
        return False, "before_start_chunk_index"
    if int(cli.frame_step) > 1 and ((int(chunk_idx) - int(cli.start_chunk_index)) % int(cli.frame_step) != 0):
        return False, "frame_step_skip"
    if float(uncovered_ratio) < float(cli.min_uncovered_ratio):
        return False, "below_min_uncovered_ratio"
    if int(cli.max_residual_frames) >= 0 and int(emitted_count) >= int(cli.max_residual_frames):
        return False, "max_residual_frames_reached"
    return True, "selected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--rgb-root", default=str(DEFAULT_RGB_ROOT))
    parser.add_argument("--inherited-birth-records", required=True)
    parser.add_argument("--handoff-replay-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=106)
    parser.add_argument("--model-dtype", default="bf16", choices=["", "fp32", "float32", "bf16", "bfloat16", "fp16", "float16"])
    parser.add_argument("--min-uncovered-ratio", type=float, default=0.0)
    parser.add_argument("--start-chunk-index", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--max-residual-frames", type=int, default=-1)
    parser.add_argument(
        "--selected-chunk-indices",
        default="",
        help="Optional comma-separated chunk_frame_index allow-list for residual birth.",
    )
    parser.add_argument(
        "--residual-input-role",
        default="inherited_handoff_replay",
        help="Audit label for the label summary used to compute residual uncovered.",
    )
    parser.add_argument("--residual-mode", choices=["gap", "frame0_twostage", "hybrid_frame0_gap"], default="gap")
    parser.add_argument(
        "--hybrid-frame0-mode",
        choices=["frame0_twostage", "frame0_stage2_only"],
        default="frame0_twostage",
        help="Objectness mode used by hybrid_frame0_gap for the first selected frame and twostage chunk overrides.",
    )
    parser.add_argument(
        "--twostage-chunk-indices",
        default="",
        help=(
            "Optional comma-separated chunk_frame_index list. In hybrid_frame0_gap mode, "
            "these selected chunks use frame0_twostage instead of gap."
        ),
    )
    parser.add_argument("--gap-max-points", type=int, default=None)
    parser.add_argument("--gap-min-component-area", type=int, default=None)
    parser.add_argument("--gap-base-points-per-component", type=int, default=None)
    parser.add_argument("--gap-area-per-extra-point", type=int, default=None)
    parser.add_argument("--gap-max-points-per-component", type=int, default=None)
    parser.add_argument(
        "--gap-choice-policy",
        choices=["largest_valid_mask_per_point", "smallest_valid_mask_per_point"],
        default=None,
    )
    parser.add_argument("--gap-iou-threshold", type=float, default=None)
    parser.add_argument("--gap-stability-threshold", type=float, default=None)
    parser.add_argument("--gap-apply-box-nms", action="store_true", default=None)
    parser.add_argument("--gap-box-nms-thresh", type=float, default=None)
    parser.add_argument("--gap-nms-score-type", choices=["pred_iou", "stability"], default="stability")
    parser.add_argument(
        "--max-birth-mask-area",
        type=int,
        default=0,
        help="Reject residual birth masks with pixel area above this value; 0 disables.",
    )
    parser.add_argument(
        "--max-birth-mask-area-ratio",
        type=float,
        default=0.0,
        help="Reject residual birth masks with mask_area / image_area above this value; 0 disables.",
    )
    parser.add_argument(
        "--max-birth-mask-uncovered-ratio",
        type=float,
        default=0.0,
        help="Reject residual birth masks with mask_area / uncovered_area above this value; 0 disables.",
    )
    parser.add_argument(
        "--frame0-child-split-mode",
        choices=["none", "append", "replace_parent", "parent_minus_children"],
        default="none",
        help="Optional non-oracle child-prompt split for high-area frame0 objectness masks.",
    )
    parser.add_argument(
        "--frame0-child-target-source",
        choices=["stage2_only", "stage1_only", "all"],
        default="stage2_only",
    )
    parser.add_argument("--frame0-child-min-parent-area-ratio", type=float, default=0.0)
    parser.add_argument("--frame0-child-max-parent-area-ratio", type=float, default=1.0)
    parser.add_argument("--frame0-child-max-parent-count", type=int, default=8)
    parser.add_argument(
        "--frame0-child-parent-raw-indices",
        default="",
        help="Optional comma-separated raw frame0 parent mask indices to allow for child split; empty uses source/area rules.",
    )
    parser.add_argument("--frame0-child-points-per-parent", type=int, default=8)
    parser.add_argument(
        "--frame0-child-max-children-per-parent",
        type=int,
        default=0,
        help="Keep at most this many child masks per targeted frame0 parent after SAM2 decode; 0 disables.",
    )
    parser.add_argument("--frame0-child-min-component-area", type=int, default=128)
    parser.add_argument(
        "--frame0-child-largest-component-only",
        action="store_true",
        default=False,
        help="Default off. Replace each accepted frame0 child mask with its largest connected component.",
    )
    parser.add_argument(
        "--frame0-child-min-largest-component-ratio",
        type=float,
        default=0.0,
        help="Reject frame0 child masks whose largest connected component / mask area is below this ratio; 0 disables.",
    )
    parser.add_argument("--frame0-child-parent-residual-min-area", type=int, default=128)
    parser.add_argument(
        "--frame0-child-negative-points-per-positive",
        type=int,
        default=0,
        help="Add this many parent-internal negative points to each frame0 child prompt; 0 preserves positive-only behavior.",
    )
    parser.add_argument(
        "--frame0-child-negative-min-distance-px",
        type=int,
        default=24,
        help="Minimum pixel distance between a child positive point and sampled parent-internal negative points.",
    )
    parser.add_argument(
        "--frame0-child-choice-policy",
        choices=["smallest_valid_mask_per_point", "largest_valid_mask_per_point"],
        default="smallest_valid_mask_per_point",
    )
    parser.add_argument("--frame0-child-iou-threshold", type=float, default=0.8)
    parser.add_argument("--frame0-child-stability-threshold", type=float, default=0.8)
    parser.add_argument("--frame0-child-apply-box-nms", action="store_true", default=False)
    parser.add_argument("--frame0-child-nms-score-type", choices=["pred_iou", "stability"], default="stability")
    parser.add_argument(
        "--repair-birth-defer-mode",
        choices=["off", "overlap"],
        default="off",
        help="Default off. When overlap, classify residual masks against handoff labels before assigning new IDs.",
    )
    parser.add_argument("--repair-overlap-coeff", type=float, default=0.55)
    parser.add_argument("--duplicate-suppress-overlap-coeff", type=float, default=0.90)
    parser.add_argument("--birth-max-overlap-coeff", type=float, default=0.25)
    parser.add_argument("--repair-birth-defer-min-area", type=int, default=0)
    parser.add_argument("--ambiguous-overlap-action", choices=["defer", "birth_new", "noise"], default="defer")
    parser.add_argument(
        "--temporal-residual-repair-mode",
        choices=["off", "mask_overlap"],
        default="off",
        help=(
            "Default off. When mask_overlap, residual masks that would become new births "
            "can reuse an earlier residual obj_id if their masks persist by non-oracle image-space overlap."
        ),
    )
    parser.add_argument("--temporal-residual-min-overlap", type=float, default=0.85)
    parser.add_argument("--temporal-residual-min-area", type=int, default=1024)
    parser.add_argument(
        "--temporal-residual-max-area-ratio",
        type=float,
        default=8.0,
        help="Reject temporal matches when max(area_a, area_b) / min(area_a, area_b) exceeds this; <=0 disables.",
    )
    parser.add_argument(
        "--temporal-residual-window-chunks",
        type=int,
        default=0,
        help="Use only previous residual masks within this chunk-frame distance; 0 uses all previous selected frames.",
    )
    parser.add_argument(
        "--temporal-residual-min-target-age-chunks",
        type=int,
        default=0,
        help=(
            "Require the matched residual target to be at least this many chunk-frame steps old before repair. "
            "0 preserves immediate update behavior."
        ),
    )
    parser.add_argument(
        "--temporal-residual-young-match-action",
        choices=["repair_existing", "noise", "defer", "birth_new"],
        default="repair_existing",
        help="Action to apply when a temporal match is found but the matched target is younger than the age gate.",
    )
    parser.add_argument("--source", default="v106_residual_gap_birth_nonoracle")
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    selected_text = str(cli.selected_chunk_indices).strip()
    if selected_text:
        cli.selected_chunk_index_set = {int(part) for part in selected_text.split(",") if part.strip()}
    else:
        cli.selected_chunk_index_set = None
    twostage_text = str(cli.twostage_chunk_indices).strip()
    if twostage_text:
        cli.twostage_chunk_index_set = {int(part) for part in twostage_text.split(",") if part.strip()}
    else:
        cli.twostage_chunk_index_set = set()
    parent_index_text = str(cli.frame0_child_parent_raw_indices).strip()
    if parent_index_text:
        cli.frame0_child_parent_raw_index_set = {
            int(part) for part in parent_index_text.split(",") if part.strip()
        }
    else:
        cli.frame0_child_parent_raw_index_set = None
    config_path = _resolve(cli.config)
    rgb_root = _resolve(cli.rgb_root)
    inherited_path = _resolve(cli.inherited_birth_records)
    replay_summary_path = _resolve(cli.handoff_replay_summary)
    output_root = _resolve(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    inherited = _read_json(inherited_path)
    replay_summary = _read_json(replay_summary_path)
    inherited_rows = [dict(row) for row in inherited.get("rows", [])]
    frame_ids = [int(v) for v in replay_summary.get("frame_ids", [])]
    if not frame_ids:
        frame_ids = [int(v) for v in inherited.get("frame_ids", [])]
    if [int(v) for v in inherited.get("frame_ids", [])] and [int(v) for v in inherited.get("frame_ids", [])] != frame_ids:
        raise ValueError("inherited birth records and handoff replay summary frame_ids differ")
    if str(inherited.get("scene_id")) != str(cli.scene_id):
        raise ValueError(f"inherited scene_id={inherited.get('scene_id')} does not match {cli.scene_id}")
    if str(replay_summary.get("scene_id")) != str(cli.scene_id):
        raise ValueError(f"replay scene_id={replay_summary.get('scene_id')} does not match {cli.scene_id}")

    records_by_chunk = {int(row["chunk_frame_index"]): row for row in replay_summary.get("records", [])}
    missing_chunks = [idx for idx in range(len(frame_ids)) if idx not in records_by_chunk]
    if missing_chunks:
        raise ValueError(f"handoff replay summary missing chunk records: {missing_chunks[:8]}")

    args = _make_baseline_args(config_path, cli, frame_ids)
    models = setup_models(args)
    segmentor = models["segmentor"]

    next_obj_id = max([int(row["obj_id"]) for row in inherited_rows], default=-1) + 1
    merged_rows: list[dict[str, Any]] = []
    for row in inherited_rows:
        copied = dict(row)
        copied.setdefault("phase5_role", "inherited")
        copied.setdefault("valid_v106_residual_candidate_input", True)
        merged_rows.append(copied)

    residual_rows: list[dict[str, Any]] = []
    per_frame_records: list[dict[str, Any]] = []
    total_segmentation_sec = 0.0
    selected_count = 0
    repair_birth_defer_records: list[dict[str, Any]] = []
    repair_birth_defer_action_counts: dict[str, int] = {}
    temporal_residual_repair_records: list[dict[str, Any]] = []
    temporal_residual_registry: list[dict[str, Any]] = []
    source = str(cli.source)
    birth_mask_root = output_root / "masks"

    for chunk_idx, frame_id in enumerate(frame_ids):
        rgb_path = rgb_root / str(cli.scene_id) / "color" / f"{int(frame_id)}.jpg"
        rgb = read_rgb(rgb_path)
        h, w = rgb.shape[:2]
        replay_row = records_by_chunk[int(chunk_idx)]
        label_path = _resolve(str(replay_row["label_path"]))
        label = _load_label(label_path, h, w)
        current_ids, current_masks = _label_to_masks(label)
        uncovered = uncovered_from_masks(current_masks, h, w)
        uncovered_ratio = float(np.count_nonzero(uncovered)) / float(uncovered.size)
        should_run, skip_reason = _selected_frame(chunk_idx, uncovered_ratio, cli, selected_count)

        if should_run:
            seg_t0 = time.time()
            effective_mode = str(cli.residual_mode)
            if effective_mode == "hybrid_frame0_gap":
                effective_mode = (
                    str(cli.hybrid_frame0_mode)
                    if int(selected_count) == 0 or int(chunk_idx) in set(cli.twostage_chunk_index_set)
                    else "gap"
                )
            if effective_mode == "gap":
                gap_seed = stable_seed(int(cli.seed), str(cli.scene_id), int(frame_id), int(args.gap_num_pts_active), source)
                gap_points, sampling_meta = sample_component_adaptive_points_yx(
                    uncovered,
                    max_points=_override_int(cli.gap_max_points, int(args.gap_max_points)),
                    min_component_area=_override_int(cli.gap_min_component_area, int(args.gap_min_component_area)),
                    base_points_per_component=_override_int(
                        cli.gap_base_points_per_component,
                        int(args.gap_base_points_per_component),
                    ),
                    area_per_extra_point=_override_int(cli.gap_area_per_extra_point, int(args.gap_area_per_extra_point)),
                    max_points_per_component=_override_int(
                        cli.gap_max_points_per_component,
                        int(args.gap_max_points_per_component),
                    ),
                    seed=gap_seed,
                )
                if int(gap_points.shape[0]) > 0:
                    gap_masks, gap_stats = run_sam2_point_segment_choice(
                        segmentor,
                        rgb,
                        points_yx=gap_points,
                        region_mask=uncovered,
                        points_per_batch=int(args.points_per_batch),
                        choice_policy=str(cli.gap_choice_policy or args.gap_choice_policy),
                        iou_threshold=_override_float(cli.gap_iou_threshold, float(args.gap_iou_threshold)),
                        stability_threshold=_override_float(
                            cli.gap_stability_threshold,
                            float(args.gap_stability_threshold),
                        ),
                        stability_score_offset=float(args.stability_score_offset),
                        model_mask_thresh=float(args.model_mask_thresh),
                        box_nms_thresh=_override_float(cli.gap_box_nms_thresh, float(args.box_nms_thresh)),
                        empty_ratio=float(args.empty_ratio),
                        apply_box_nms=bool(args.gap_apply_box_nms if cli.gap_apply_box_nms is None else cli.gap_apply_box_nms),
                        nms_score_type=str(cli.gap_nms_score_type),
                    )
                else:
                    gap_masks = np.zeros((0, h, w), dtype=bool)
                    gap_stats = {
                        "choice_policy": str(cli.gap_choice_policy or args.gap_choice_policy),
                        "raw_multimask_option_count": 0,
                        "prompt_with_good_mask_count": 0,
                        "pre_nms_mask_count": 0,
                        "post_disjoint_mask_count": 0,
                        "apply_box_nms": bool(args.gap_apply_box_nms if cli.gap_apply_box_nms is None else cli.gap_apply_box_nms),
                        "nms_score_type": str(cli.gap_nms_score_type),
                    }
                gap_mask_source_records = [
                    {
                        "stage": "gap",
                        "stage_mask_index": int(idx),
                        "original_raw_mask_index": int(idx),
                        "frame0_child_split_role": "not_applicable",
                    }
                    for idx in range(int(gap_masks.shape[0]))
                ]
            elif effective_mode in {"frame0_twostage", "frame0_stage2_only"}:
                s1_seed = stable_seed(
                    int(cli.seed),
                    str(cli.scene_id),
                    int(frame_id),
                    int(args.stage1_num_pts),
                    str(args.stage1_point_mode),
                    "v106-residual-frame0-stage1",
                )
                if effective_mode == "frame0_twostage":
                    s1_points, s1_point_meta = make_points_yx_torch(
                        int(args.stage1_num_pts),
                        s1_seed,
                        str(args.stage1_point_mode),
                    )
                    stage1_t0 = time.time()
                    stage1_masks, stage1_stats = run_sam2_point_segment_choice(
                        segmentor,
                        rgb,
                        points_yx=s1_points,
                        region_mask=uncovered,
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
                    stage1_sec = time.time() - stage1_t0
                    if current_masks.size and stage1_masks.size:
                        after_stage1 = np.concatenate([current_masks, stage1_masks], axis=0)
                    elif current_masks.size:
                        after_stage1 = current_masks
                    else:
                        after_stage1 = stage1_masks
                    stage2_region = uncovered_from_masks(after_stage1, h, w)
                else:
                    s1_point_meta = {
                        "sampler": "disabled",
                        "reason": "frame0_stage2_only",
                        "point_count": 0,
                    }
                    stage1_masks = np.zeros((0, h, w), dtype=bool)
                    stage1_stats = {
                        "mode": "disabled",
                        "reason": "frame0_stage2_only",
                        "raw_multimask_option_count": 0,
                        "prompt_with_good_mask_count": 0,
                        "pre_nms_mask_count": 0,
                        "post_disjoint_mask_count": 0,
                    }
                    stage1_sec = 0.0
                    stage2_region = uncovered
                s2_seed = stable_seed(
                    int(cli.seed),
                    str(cli.scene_id),
                    int(frame_id),
                    int(args.stage2_num_pts),
                    "v106-residual-frame0-stage2",
                )
                s2_points = sample_points_from_mask_yx(
                    stage2_region,
                    int(args.stage2_num_pts),
                    s2_seed,
                    inner_margin=int(args.gap_inner_margin),
                )
                stage2_t0 = time.time()
                if int(s2_points.shape[0]) > 0:
                    stage2_masks, stage2_stats = run_sam2_point_segment_choice(
                        segmentor,
                        rgb,
                        points_yx=s2_points,
                        region_mask=stage2_region,
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
                stage2_sec = time.time() - stage2_t0
                if stage1_masks.size and stage2_masks.size:
                    gap_masks = np.concatenate([stage1_masks, stage2_masks], axis=0)
                elif stage1_masks.size:
                    gap_masks = stage1_masks
                else:
                    gap_masks = stage2_masks
                gap_mask_source_records = []
                for idx in range(int(stage1_masks.shape[0])):
                    gap_mask_source_records.append(
                        {
                            "stage": "stage1",
                            "stage_mask_index": int(idx),
                            "original_raw_mask_index": int(idx),
                        }
                    )
                for idx in range(int(stage2_masks.shape[0])):
                    gap_mask_source_records.append(
                        {
                            "stage": "stage2",
                            "stage_mask_index": int(idx),
                            "original_raw_mask_index": int(stage1_masks.shape[0] + idx),
                        }
                    )
                gap_masks, gap_mask_source_records, child_split_summary = _apply_frame0_child_split(
                    segmentor=segmentor,
                    rgb=rgb,
                    masks=gap_masks,
                    source_records=gap_mask_source_records,
                    args=args,
                    cli=cli,
                    frame_id=int(frame_id),
                )
                sampling_meta = {
                    "sampler": str(effective_mode),
                    "stage1_point_sampling": s1_point_meta,
                    "stage2_point_count": int(s2_points.shape[0]),
                    "uncovered_ratio_after_stage1": float(np.count_nonzero(stage2_region)) / float(stage2_region.size),
                }
                gap_stats = {
                    "mode": str(effective_mode),
                    "stage1_runtime_sec": float(stage1_sec),
                    "stage2_runtime_sec": float(stage2_sec),
                    "stage1_mask_count": int(stage1_masks.shape[0]),
                    "stage2_mask_count": int(stage2_masks.shape[0]),
                    "stage1_stats": stage1_stats,
                    "stage2_stats": stage2_stats,
                    "frame0_child_split": child_split_summary,
                }
            else:
                raise ValueError(f"unsupported residual_mode={cli.residual_mode}")
            pre_admission_mask_count = int(gap_masks.shape[0])
            gap_masks, accepted_mask_meta, admission_summary = _apply_birth_admission_policy(
                gap_masks,
                uncovered=uncovered,
                cli=cli,
            )
            gap_sec = time.time() - seg_t0
            total_segmentation_sec += gap_sec
            accepted_gap_mask_count = int(gap_masks.shape[0])
            frame_rbd_records: list[dict[str, Any]] = []
            scheduled_items: list[dict[str, Any]] = []
            scheduled_by_obj: dict[int, int] = {}
            frame_rbd_action_counts: dict[str, int] = {}
            for candidate_index, (candidate_mask, mask_meta) in enumerate(
                zip(gap_masks.astype(bool), accepted_mask_meta, strict=False)
            ):
                raw_index = int(mask_meta["raw_mask_index"])
                source_record = {}
                if 0 <= raw_index < len(gap_mask_source_records):
                    source_record = dict(gap_mask_source_records[raw_index])
                classification = _classify_repair_birth_defer_candidate(
                    mask=candidate_mask,
                    label=label,
                    cli=cli,
                )
                action = str(classification["action"])
                temporal_match = None
                if action == "birth_new":
                    temporal_match = _best_temporal_residual_match(
                        mask=candidate_mask,
                        registry=temporal_residual_registry,
                        chunk_frame_index=int(chunk_idx),
                        cli=cli,
                    )
                    if temporal_match is not None:
                        target_age = int(chunk_idx) - int(temporal_match["target_chunk_frame_index"])
                        temporal_match = dict(temporal_match)
                        temporal_match["target_age_chunks"] = int(target_age)
                        age_gate = int(cli.temporal_residual_min_target_age_chunks)
                        if age_gate > 0 and target_age < age_gate:
                            young_action = str(cli.temporal_residual_young_match_action)
                            classification = dict(classification)
                            classification.update(
                                {
                                    "action": young_action,
                                    "target_obj_id": (
                                        int(temporal_match["target_obj_id"])
                                        if young_action == "repair_existing"
                                        else None
                                    ),
                                    "reason": "temporal_residual_match_target_too_young",
                                    "temporal_residual_repair": temporal_match,
                                    "temporal_residual_age_gate": {
                                        "min_target_age_chunks": int(age_gate),
                                        "target_age_chunks": int(target_age),
                                        "young_match_action": young_action,
                                    },
                                }
                            )
                            action = young_action
                        else:
                            classification = dict(classification)
                            classification.update(
                                {
                                    "action": "repair_existing",
                                    "target_obj_id": int(temporal_match["target_obj_id"]),
                                    "reason": "temporal_residual_mask_overlap_reuse",
                                    "temporal_residual_repair": temporal_match,
                                }
                            )
                            action = "repair_existing"
                        if action in {"noise", "defer"}:
                            temporal_residual_repair_records.append(
                                {
                                    "chunk_frame_index": int(chunk_idx),
                                    "frame_id": int(frame_id),
                                    "candidate_index": int(candidate_index),
                                    "raw_mask_index": int(mask_meta["raw_mask_index"]),
                                    "applied_action": action,
                                    "age_gate_blocked_repair": True,
                                    **temporal_match,
                                }
                            )
                        elif action == "repair_existing":
                            temporal_residual_repair_records.append(
                                {
                                    "chunk_frame_index": int(chunk_idx),
                                    "frame_id": int(frame_id),
                                    "candidate_index": int(candidate_index),
                                    "raw_mask_index": int(mask_meta["raw_mask_index"]),
                                    "applied_action": action,
                                    "age_gate_blocked_repair": False,
                                    **temporal_match,
                                }
                            )
                frame_rbd_action_counts[action] = int(frame_rbd_action_counts.get(action, 0)) + 1
                repair_birth_defer_action_counts[action] = int(repair_birth_defer_action_counts.get(action, 0)) + 1
                audit_record = {
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "candidate_index": int(candidate_index),
                    "raw_mask_index": int(raw_index),
                    "source_record": source_record,
                    "mask_area_ratio": float(mask_meta["mask_area_ratio"]),
                    "mask_uncovered_ratio": float(mask_meta["mask_uncovered_ratio"]),
                    **classification,
                }
                frame_rbd_records.append(audit_record)
                repair_birth_defer_records.append(audit_record)
                scheduled_obj_id: int | None = None
                if action == "birth_new":
                    scheduled_obj_id = int(next_obj_id)
                    next_obj_id += 1
                elif action == "repair_existing":
                    scheduled_obj_id = int(classification["target_obj_id"])
                if scheduled_obj_id is None:
                    continue
                item_index = scheduled_by_obj.get(int(scheduled_obj_id))
                if item_index is None:
                    scheduled_by_obj[int(scheduled_obj_id)] = len(scheduled_items)
                    scheduled_items.append(
                        {
                            "obj_id": int(scheduled_obj_id),
                            "mask": candidate_mask.astype(bool),
                            "action": action,
                            "source_record": source_record,
                            "mask_meta": dict(mask_meta),
                            "classification_records": [audit_record],
                        }
                    )
                else:
                    item = scheduled_items[int(item_index)]
                    item["mask"] = item["mask"].astype(bool) | candidate_mask.astype(bool)
                    item["classification_records"].append(audit_record)
                    item["action"] = "repair_existing" if action == "repair_existing" else str(item["action"])

            if scheduled_items:
                for item in scheduled_items:
                    item_area = int(np.count_nonzero(item["mask"].astype(bool)))
                    if item_area >= int(cli.temporal_residual_min_area):
                        temporal_residual_registry.append(
                            {
                                "obj_id": int(item["obj_id"]),
                                "chunk_frame_index": int(chunk_idx),
                                "frame_id": int(frame_id),
                                "area": int(item_area),
                                "mask": item["mask"].astype(bool),
                                "source": str(item["action"]),
                            }
                        )

            if scheduled_items:
                gap_ids = np.asarray([int(item["obj_id"]) for item in scheduled_items], dtype=np.int64)
                scheduled_masks = np.stack([item["mask"].astype(bool) for item in scheduled_items], axis=0)
            else:
                gap_ids = np.zeros((0,), dtype=np.int64)
                scheduled_masks = np.zeros((0, h, w), dtype=bool)
            gap_masks = scheduled_masks
            before = len(residual_rows)
            dump_birth_masks(
                birth_mask_root,
                residual_rows,
                scene_id=str(cli.scene_id),
                chunk_frame_index=int(chunk_idx),
                frame_id=int(frame_id),
                source=source,
                obj_ids=gap_ids,
                masks=gap_masks,
            )
            for row, scheduled_item in zip(residual_rows[before:], scheduled_items, strict=False):
                source_record = dict(scheduled_item["source_record"])
                mask_meta = dict(scheduled_item["mask_meta"])
                classification_records = list(scheduled_item["classification_records"])
                scheduled_action = str(scheduled_item["action"])
                row["source"] = f"{source}_{scheduled_action}"
                row["phase5_role"] = "repair_existing" if scheduled_action == "repair_existing" else "birth_new"
                row["global_id"] = int(row["obj_id"]) + 1
                row["nonoracle"] = True
                row["residual_gap_input_label_path"] = _rel(label_path)
                row["residual_gap_uncovered_ratio_before_birth"] = float(uncovered_ratio)
                row["residual_birth_mask_source"] = source_record
                row["repair_birth_defer_mode"] = str(cli.repair_birth_defer_mode)
                row["repair_birth_defer_action"] = scheduled_action
                row["repair_birth_defer_merged_candidate_count"] = int(len(classification_records))
                row["repair_birth_defer_classification_records"] = classification_records[:8]
                row["frame0_child_split_role"] = str(source_record.get("frame0_child_split_role", "not_applicable"))
                if "frame0_child_parent_raw_mask_index" in source_record:
                    row["frame0_child_parent_raw_mask_index"] = int(source_record["frame0_child_parent_raw_mask_index"])
                if "frame0_child_parent_stage" in source_record:
                    row["frame0_child_parent_stage"] = str(source_record["frame0_child_parent_stage"])
                if "frame0_child_parent_area_ratio" in source_record:
                    row["frame0_child_parent_area_ratio"] = float(source_record["frame0_child_parent_area_ratio"])
                if "frame0_child_index" in source_record:
                    row["frame0_child_index"] = int(source_record["frame0_child_index"])
                if "frame0_child_count" in source_record:
                    row["frame0_child_count"] = int(source_record["frame0_child_count"])
                row["birth_admission_policy"] = {
                    "enabled": bool(admission_summary["enabled"]),
                    "max_birth_mask_area": int(cli.max_birth_mask_area),
                    "max_birth_mask_area_ratio": float(cli.max_birth_mask_area_ratio),
                    "max_birth_mask_uncovered_ratio": float(cli.max_birth_mask_uncovered_ratio),
                }
                row["birth_admission_raw_mask_index"] = int(mask_meta["raw_mask_index"])
                row["birth_admission_mask_area_ratio"] = float(mask_meta["mask_area_ratio"])
                row["birth_admission_mask_uncovered_ratio"] = float(mask_meta["mask_uncovered_ratio"])
            gap_stats["birth_admission"] = admission_summary
            gap_stats["pre_admission_mask_count"] = int(pre_admission_mask_count)
            gap_stats["accepted_gap_mask_count"] = int(accepted_gap_mask_count)
            gap_stats["scheduled_gap_mask_count"] = int(gap_masks.shape[0])
            gap_stats["repair_birth_defer"] = {
                "mode": str(cli.repair_birth_defer_mode),
                "action_counts": {k: int(v) for k, v in sorted(frame_rbd_action_counts.items())},
                "candidate_count": int(len(frame_rbd_records)),
                "scheduled_row_count": int(len(scheduled_items)),
                "records": frame_rbd_records[:32],
            }
            selected_count += 1
        else:
            sampling_meta = {
                "sampler": "component_adaptive",
                "point_count": 0,
                "skip_reason": skip_reason,
            }
            gap_stats = {
                "choice_policy": str(cli.gap_choice_policy or args.gap_choice_policy),
                "raw_multimask_option_count": 0,
                "prompt_with_good_mask_count": 0,
                "pre_nms_mask_count": 0,
                "post_disjoint_mask_count": 0,
                "apply_box_nms": bool(args.gap_apply_box_nms if cli.gap_apply_box_nms is None else cli.gap_apply_box_nms),
                "nms_score_type": str(cli.gap_nms_score_type),
            }
            gap_masks = np.zeros((0, h, w), dtype=bool)
            gap_sec = 0.0
            admission_summary = {
                "enabled": bool(_birth_admission_enabled(cli)),
                "max_birth_mask_area": int(cli.max_birth_mask_area),
                "max_birth_mask_area_ratio": float(cli.max_birth_mask_area_ratio),
                "max_birth_mask_uncovered_ratio": float(cli.max_birth_mask_uncovered_ratio),
                "pre_admission_mask_count": 0,
                "post_admission_mask_count": 0,
                "filtered_birth_mask_count": 0,
                "filtered_by_reason": {},
            }

        frame_record = {
            "chunk_frame_index": int(chunk_idx),
            "frame_id": int(frame_id),
            "rgb_path": _rel(rgb_path),
            "input_handoff_label_path": _rel(label_path),
            "handoff_visible_id_count": int(current_ids.shape[0]),
            "handoff_foreground_ratio": float(np.count_nonzero(label) / float(label.size)),
            "uncovered_ratio_before_birth": float(uncovered_ratio),
            "selected_for_residual_birth": bool(should_run),
            "effective_residual_mode": str(effective_mode) if should_run else None,
            "skip_reason": str(skip_reason),
            "gap_runtime_sec": float(gap_sec),
            "gap_mask_count": int(gap_masks.shape[0]),
            "birth_admission": admission_summary,
            "gap_sampling": sampling_meta,
            "gap_stats": gap_stats,
        }
        per_frame_records.append(frame_record)
        print(json.dumps(frame_record, ensure_ascii=True), flush=True)

    merged_rows.extend(residual_rows)
    merged_rows.sort(key=lambda row: (int(row["chunk_frame_index"]), int(row["obj_id"])))
    payload = {
        "schema_version": "stream4d_v106_phase9_residual_gap_birth_bank_v1",
        "valid_v106_method_candidate": True,
        "diagnostic_only": False,
        "nonoracle": True,
        "known_limitation": (
            "Residual uncovered is computed from the supplied replay labels. "
            "Use residual_input_role to audit whether this was inherited-only handoff replay "
            "or a later closed-loop/final-repair replay."
        ),
        "residual_input_role": str(cli.residual_input_role),
        "scene_id": str(cli.scene_id),
        "frame_ids": [int(v) for v in frame_ids],
        "config_path": _rel(config_path),
        "config_sha256": sha256_file(config_path),
        "inherited_birth_records": _rel(inherited_path),
        "inherited_birth_records_sha256": sha256_file(inherited_path),
        "handoff_replay_summary": _rel(replay_summary_path),
        "handoff_replay_summary_sha256": sha256_file(replay_summary_path),
        "source": source,
        "seed": int(cli.seed),
        "model_dtype": str(cli.model_dtype),
        "residual_mode": str(cli.residual_mode),
        "gap_policy": {
            "sampler": "component_adaptive",
            "choice_policy": str(cli.gap_choice_policy or args.gap_choice_policy),
            "pred_iou_thresh": _override_float(cli.gap_iou_threshold, float(args.gap_iou_threshold)),
            "stability_score_thresh": _override_float(
                cli.gap_stability_threshold,
                float(args.gap_stability_threshold),
            ),
            "apply_box_nms": bool(args.gap_apply_box_nms if cli.gap_apply_box_nms is None else cli.gap_apply_box_nms),
            "box_nms_thresh": _override_float(cli.gap_box_nms_thresh, float(args.box_nms_thresh)),
            "nms_score_type": str(cli.gap_nms_score_type),
            "max_points": _override_int(cli.gap_max_points, int(args.gap_max_points)),
            "min_component_area": _override_int(cli.gap_min_component_area, int(args.gap_min_component_area)),
            "base_points_per_component": _override_int(
                cli.gap_base_points_per_component,
                int(args.gap_base_points_per_component),
            ),
            "area_per_extra_point": _override_int(cli.gap_area_per_extra_point, int(args.gap_area_per_extra_point)),
            "max_points_per_component": _override_int(
                cli.gap_max_points_per_component,
                int(args.gap_max_points_per_component),
            ),
            "points_per_batch": int(args.points_per_batch),
        },
        "birth_admission_policy": {
            "enabled": bool(_birth_admission_enabled(cli)),
            "max_birth_mask_area": int(cli.max_birth_mask_area),
            "max_birth_mask_area_ratio": float(cli.max_birth_mask_area_ratio),
            "max_birth_mask_uncovered_ratio": float(cli.max_birth_mask_uncovered_ratio),
        },
        "frame0_child_split_policy": {
            "enabled": bool(str(cli.frame0_child_split_mode) != "none"),
            "mode": str(cli.frame0_child_split_mode),
            "target_source": str(cli.frame0_child_target_source),
            "min_parent_area_ratio": float(cli.frame0_child_min_parent_area_ratio),
            "max_parent_area_ratio": float(cli.frame0_child_max_parent_area_ratio),
            "max_parent_count": int(cli.frame0_child_max_parent_count),
            "parent_raw_index_allowlist": (
                [int(v) for v in sorted(cli.frame0_child_parent_raw_index_set)]
                if cli.frame0_child_parent_raw_index_set is not None
                else []
            ),
            "points_per_parent": int(cli.frame0_child_points_per_parent),
            "max_children_per_parent": int(cli.frame0_child_max_children_per_parent),
            "negative_points_per_positive": int(cli.frame0_child_negative_points_per_positive),
            "negative_min_distance_px": int(cli.frame0_child_negative_min_distance_px),
            "min_component_area": int(cli.frame0_child_min_component_area),
            "largest_component_only": bool(cli.frame0_child_largest_component_only),
            "min_largest_component_ratio": float(cli.frame0_child_min_largest_component_ratio),
            "parent_residual_min_area": int(cli.frame0_child_parent_residual_min_area),
            "choice_policy": str(cli.frame0_child_choice_policy),
            "iou_threshold": float(cli.frame0_child_iou_threshold),
            "stability_threshold": float(cli.frame0_child_stability_threshold),
            "apply_box_nms": bool(cli.frame0_child_apply_box_nms),
            "nms_score_type": str(cli.frame0_child_nms_score_type),
        },
        "repair_birth_defer_policy": {
            "mode": str(cli.repair_birth_defer_mode),
            "repair_overlap_coeff": float(cli.repair_overlap_coeff),
            "duplicate_suppress_overlap_coeff": float(cli.duplicate_suppress_overlap_coeff),
            "birth_max_overlap_coeff": float(cli.birth_max_overlap_coeff),
            "min_area": int(cli.repair_birth_defer_min_area),
            "ambiguous_overlap_action": str(cli.ambiguous_overlap_action),
            "action_counts": {k: int(v) for k, v in sorted(repair_birth_defer_action_counts.items())},
            "candidate_count": int(len(repair_birth_defer_records)),
            "records": repair_birth_defer_records[:256],
        },
        "temporal_residual_repair_policy": {
            "mode": str(cli.temporal_residual_repair_mode),
            "enabled": bool(_temporal_residual_repair_enabled(cli)),
            "min_overlap": float(cli.temporal_residual_min_overlap),
            "min_area": int(cli.temporal_residual_min_area),
            "max_area_ratio": float(cli.temporal_residual_max_area_ratio),
            "window_chunks": int(cli.temporal_residual_window_chunks),
            "min_target_age_chunks": int(cli.temporal_residual_min_target_age_chunks),
            "young_match_action": str(cli.temporal_residual_young_match_action),
            "match_count": int(len(temporal_residual_repair_records)),
            "records": temporal_residual_repair_records[:256],
        },
        "frame_selection": {
            "min_uncovered_ratio": float(cli.min_uncovered_ratio),
            "start_chunk_index": int(cli.start_chunk_index),
            "frame_step": int(cli.frame_step),
            "max_residual_frames": int(cli.max_residual_frames),
            "selected_chunk_indices": (
                [int(v) for v in sorted(cli.selected_chunk_index_set)]
                if cli.selected_chunk_index_set is not None
                else []
            ),
            "twostage_chunk_indices": [int(v) for v in sorted(cli.twostage_chunk_index_set)],
        },
        "runtime_tuning": models.get("runtime_tuning", {}),
        "total_residual_segmentation_runtime_sec": float(total_segmentation_sec),
        "inherited_row_count": int(len(inherited_rows)),
        "residual_row_count": int(len(residual_rows)),
        "filtered_residual_mask_count": int(
            sum(int(row.get("birth_admission", {}).get("filtered_birth_mask_count", 0)) for row in per_frame_records)
        ),
        "row_count": int(len(merged_rows)),
        "per_frame_records": per_frame_records,
        "rows": merged_rows,
    }
    out_path = output_root / "residual_gap_birth_records.json"
    write_json(out_path, payload)
    summary = {
        "schema_version": "stream4d_v106_phase9_residual_gap_birth_bank_summary_v1",
        "valid_v106_method_candidate": True,
        "diagnostic_only": False,
        "nonoracle": True,
        "birth_records": _rel(out_path),
        "birth_records_sha256": sha256_file(out_path),
        "scene_id": str(cli.scene_id),
        "frame_count": len(frame_ids),
        "inherited_row_count": int(len(inherited_rows)),
        "residual_row_count": int(len(residual_rows)),
        "row_count": int(len(merged_rows)),
        "selected_frame_count": int(selected_count),
        "total_residual_segmentation_runtime_sec": float(total_segmentation_sec),
        "birth_admission_policy": payload["birth_admission_policy"],
        "frame0_child_split_policy": payload["frame0_child_split_policy"],
        "repair_birth_defer_policy": payload["repair_birth_defer_policy"],
        "temporal_residual_repair_policy": payload["temporal_residual_repair_policy"],
        "filtered_residual_mask_count": int(payload["filtered_residual_mask_count"]),
        "residual_mode": str(cli.residual_mode),
        "residual_input_role": str(cli.residual_input_role),
        "frame_selection": payload["frame_selection"],
        "known_limitation": payload["known_limitation"],
    }
    write_json(output_root / "residual_gap_birth_bank_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
