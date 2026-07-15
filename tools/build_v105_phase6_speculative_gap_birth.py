#!/usr/bin/env python3
"""Phase6 speculative-gap birth probe for Stream4D v105.

This runner deliberately avoids the older SGQ pipeline defaults that offload
video/state to CPU. It starts from frozen frame0 seeds, uses AllTracker
core/envelope masks as an approximate future-coverage provider, decodes SAM2
birth masks from proxy-gap prompts, and propagates accepted births with the
Phase5 GPU-resident video feature bank path.

Reference X0/X1 labels are diagnostics only. They are never used to create
candidate pixels, prompts, or accepted masks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "v105" / "baseline_chunk_table" / "baseline_x_gapadaptive_sam2.generated.yaml"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    add_masks_to_stream_state,
    infer_stream_frame,
    load_config,
    make_args,
    propagate_new_masks_chunked,
    run_sam2_point_segment_choice,
    sample_component_adaptive_points_yx,
    setup_models,
    uncovered_from_masks,
)
from tools.audit_v105_4dpm_largest_tracking_baseline import (  # noqa: E402
    disjoin_keep_order,
    label_from_id_masks,
    make_numeric_frame_dir,
    make_sheet_grid,
    write_video,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    annotate_frame,
    disjoin_smallest_first,
    mask_stats,
    overlay_label,
    parse_frame_ids,
    read_rgb,
    sha256_file,
    stable_seed,
)
from tools.build_v105_phase5_frozen_birth_replay import (  # noqa: E402
    empty_feature_bank_summary,
    install_video_feature_bank_patch,
    reconcile_stream_state_object_count,
    serializable_feature_bank_summary,
)


def resolve_path(path_text: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_gray_bool(path: Path, h: int, w: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    mask = image > 0
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask.astype(bool)


def make_numeric_frame_suffix_dir(frame_paths: list[Path], output_root: Path, start_idx: int) -> Path:
    frame_dir = output_root / "_numeric_frame_suffixes" / f"start_{int(start_idx):02d}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, src in enumerate(frame_paths[int(start_idx) :]):
        dst = frame_dir / f"{int(local_idx):06d}.jpg"
        os.symlink(os.path.abspath(src), dst)
    return frame_dir


def prune_stream_noncond_memory(
    state: dict[str, Any] | None,
    *,
    current_frame_idx: int,
    keep_window: int,
) -> dict[str, Any]:
    keep_window = int(keep_window)
    if state is None or keep_window < 0:
        return {"enabled": False, "keep_window": int(keep_window)}
    output_dict = state.get("output_dict", {})
    non_cond_outputs = output_dict.get("non_cond_frame_outputs", {})
    threshold = int(current_frame_idx) - int(keep_window)
    to_remove = [int(frame_idx) for frame_idx in list(non_cond_outputs.keys()) if int(frame_idx) < int(threshold)]
    before_noncond_count = int(len(non_cond_outputs))
    for frame_idx in to_remove:
        non_cond_outputs.pop(int(frame_idx), None)
        state.get("consolidated_frame_inds", {}).get("non_cond_frame_outputs", set()).discard(int(frame_idx))
        for obj_output_dict in state.get("output_dict_per_obj", {}).values():
            obj_output_dict.get("non_cond_frame_outputs", {}).pop(int(frame_idx), None)
    return {
        "enabled": True,
        "keep_window": int(keep_window),
        "current_frame_idx": int(current_frame_idx),
        "threshold_frame_idx": int(threshold),
        "removed_noncond_frame_count": int(len(to_remove)),
        "remaining_noncond_frame_count": int(len(non_cond_outputs)),
        "before_noncond_frame_count": int(before_noncond_count),
        "removed_noncond_frame_indices": to_remove[:64],
        "removed_noncond_frame_indices_truncated": bool(len(to_remove) > 64),
    }


def split_frame0_hybrid_groups(
    obj_ids: np.ndarray,
    masks: np.ndarray,
    *,
    full_topk: int,
    full_min_area: int,
) -> dict[str, Any]:
    areas = masks.reshape((int(masks.shape[0]), -1)).sum(axis=1).astype(np.int64)
    full_keep = np.zeros((int(obj_ids.size),), dtype=bool)
    full_topk = int(full_topk)
    full_min_area = int(full_min_area)
    if full_topk > 0:
        order = np.argsort(-areas, kind="stable")
        full_keep[order[: min(full_topk, int(obj_ids.size))]] = True
    if full_min_area > 0:
        full_keep |= areas >= int(full_min_area)
    full_indices = np.nonzero(full_keep)[0].astype(np.int64)
    small_indices = np.nonzero(~full_keep)[0].astype(np.int64)
    return {
        "areas": areas,
        "full_indices": full_indices,
        "small_indices": small_indices,
        "full_area_sum": int(areas[full_indices].sum()) if int(full_indices.size) else 0,
        "small_area_sum": int(areas[small_indices].sum()) if int(small_indices.size) else 0,
        "full_ids": [int(v) for v in obj_ids[full_indices].tolist()],
        "small_ids": [int(v) for v in obj_ids[small_indices].tolist()],
    }


def run_sam2_point_segment_choice_candidate_support(
    segmentor: Any,
    rgb: np.ndarray,
    *,
    points_yx: Any,
    support_mask: np.ndarray | None,
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
    support_min_area: int,
    support_min_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    try:
        from sam2.utils.amg import batched_mask_to_box, calculate_stability_score
    except Exception:
        from efficient_track_anything.utils.amg import batched_mask_to_box, calculate_stability_score
    from torchvision.ops import nms

    h, w = rgb.shape[:2]
    choice_policy = str(choice_policy)
    supported_policies = {
        "smallest_valid_mask_per_point",
        "largest_valid_mask_per_point",
        "smallest_valid_candidate_supported",
        "max_candidate_support_valid_mask_per_point",
    }
    if choice_policy not in supported_policies:
        raise ValueError(f"unsupported choice_policy={choice_policy}")
    if str(nms_score_type) not in {"pred_iou", "stability"}:
        raise ValueError(f"unsupported nms_score_type={nms_score_type}")

    segmentor.reset_predictor()
    segmentor.set_image(rgb)
    support_t = None
    if support_mask is not None:
        support_t = torch.as_tensor(support_mask.astype(bool), device="cuda")

    selected_batches = []
    selected_score_batches = []
    prompt_with_good = 0
    prompt_with_supported = 0
    raw_option_count = 0
    supported_option_count = 0
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
            mask_bool = masks > float(model_mask_thresh)
            good = (iou_predictions > float(iou_threshold)) & (stability >= float(stability_threshold))
            raw_option_count += int(good.numel())
            areas = mask_bool.sum(dim=(-1, -2), dtype=torch.int64)
            if support_t is not None:
                support_areas = (mask_bool & support_t.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2), dtype=torch.int64)
            else:
                support_areas = areas
            support_ratios = support_areas.float() / torch.clamp(areas.float(), min=1.0)
            support_good = good & (support_areas >= int(support_min_area)) & (support_ratios >= float(support_min_ratio))
            supported_option_count += int(support_good.sum().item())

            if choice_policy == "largest_valid_mask_per_point":
                score = areas.clone()
                score[~good] = -1
                chosen_idx = score.argmax(dim=1)
                has_good = good.any(dim=1)
            elif choice_policy == "smallest_valid_mask_per_point":
                score = areas.clone()
                score[~good] = torch.iinfo(torch.int64).max // 4
                chosen_idx = score.argmin(dim=1)
                has_good = good.any(dim=1)
            elif choice_policy == "smallest_valid_candidate_supported":
                score = areas.clone()
                score[~support_good] = torch.iinfo(torch.int64).max // 4
                chosen_idx = score.argmin(dim=1)
                has_good = support_good.any(dim=1)
            else:
                score = support_areas.clone()
                score[~support_good] = -1
                chosen_idx = score.argmax(dim=1)
                has_good = support_good.any(dim=1)

            prompt_with_good += int(good.any(dim=1).sum().item())
            prompt_with_supported += int(support_good.any(dim=1).sum().item())
            prompt_indices = torch.nonzero(has_good, as_tuple=False).flatten()
            if int(prompt_indices.numel()) > 0:
                selected = mask_bool[prompt_indices, chosen_idx[prompt_indices]]
                if str(nms_score_type) == "pred_iou":
                    selected_scores = iou_predictions[prompt_indices, chosen_idx[prompt_indices]].float()
                else:
                    selected_scores = stability[prompt_indices, chosen_idx[prompt_indices]].float()
                selected_batches.append(selected)
                selected_score_batches.append(selected_scores)

    if selected_batches:
        selected_t = torch.cat(selected_batches, dim=0)
        selected_scores_t = torch.cat(selected_score_batches, dim=0)
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
        "candidate_supported_option_count": int(supported_option_count),
        "prompt_with_candidate_supported_mask_count": int(prompt_with_supported),
        "support_min_area": int(support_min_area),
        "support_min_ratio": float(support_min_ratio),
        "pre_nms_mask_count": int(pre_nms_count),
        "post_disjoint_mask_count": int(disjoint_np.shape[0]),
        "apply_box_nms": bool(apply_box_nms),
        "nms_score_type": str(nms_score_type),
    }
    return disjoint_np, stats


def load_label(path: Path, h: int, w: int) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(path)
    if label.ndim == 3:
        label = label[:, :, 0]
    if label.shape[:2] != (h, w):
        label = cv2.resize(label.astype(np.uint16), (w, h), interpolation=cv2.INTER_NEAREST)
    return label.astype(np.uint16)


def load_label_paths(summary_path: Path) -> tuple[dict[int, Path], dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    labels: dict[int, Path] = {}
    for row in payload.get("records", []):
        frame_id = int(row["frame_id"])
        label_path = resolve_path(str(row["label_path"]))
        if not label_path.exists():
            raise FileNotFoundError(label_path)
        labels[frame_id] = label_path
    return labels, payload


def make_baseline_args(config_path: Path, cli: argparse.Namespace) -> SimpleNamespace:
    config = load_config(config_path)
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=cli.scene_id,
        rgb_root=cli.rgb_root,
        frame_start=cli.frame_start,
        frame_stride=cli.frame_stride,
        frame_count=cli.frame_count,
        frame_ids=cli.frame_ids,
        output_root=cli.output_root,
        seed=cli.seed,
        birth_dump_dir="",
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(cli.output_root)
    args.offload_video_to_cpu = bool(cli.offload_video_to_cpu)
    args.offload_state_to_cpu = bool(cli.offload_state_to_cpu)
    args.propagation_chunk_size = int(cli.propagation_chunk_size)
    return args


def load_frame0_seed_rows(path: Path, frame_ids: list[int], scene_id: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload_frame_ids = [int(v) for v in payload.get("frame_ids", [])]
    requested_frame_ids = [int(v) for v in frame_ids]
    frame_id_contract = "absent"
    if payload_frame_ids:
        if payload_frame_ids == requested_frame_ids:
            frame_id_contract = "full_requested_sequence"
        elif payload_frame_ids == [requested_frame_ids[0]]:
            frame_id_contract = "frame0_only_seed_bank"
        else:
            raise ValueError(
                {
                    "frame_id_mismatch_in_birth_bank": str(path),
                    "payload_frame_ids": payload_frame_ids[:16],
                    "requested_frame_ids": requested_frame_ids[:16],
                    "allowed": "full requested sequence or frame0-only seed bank",
                }
            )
    rows = []
    for row in payload.get("rows", []):
        if str(row.get("scene_id")) != str(scene_id):
            raise ValueError(f"scene mismatch in birth bank: {row.get('scene_id')} != {scene_id}")
        if str(row.get("source")) != "frame0_seed":
            continue
        if int(row.get("chunk_frame_index", -1)) != 0:
            continue
        mask_path = resolve_path(str(row["mask_path"]))
        rows.append((int(row["obj_id"]), mask_path, int(row.get("mask_area", 0))))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError(f"no frame0_seed rows in {path}")
    obj_ids = np.asarray([item[0] for item in rows], dtype=np.int64)
    masks = np.stack([load_gray_bool(item[1], h, w) for item in rows], axis=0).astype(bool)
    meta = {
        "birth_records_path": str(path),
        "birth_records_sha256": sha256_file(path),
        "birth_records_frame_id_contract": frame_id_contract,
        "birth_records_payload_frame_count": int(len(payload_frame_ids)),
        "requested_frame_count": int(len(requested_frame_ids)),
        "frame0_seed_count": int(len(rows)),
        "frame0_seed_obj_id_min": int(obj_ids.min()),
        "frame0_seed_obj_id_max": int(obj_ids.max()),
        "frame0_seed_area_total": int(np.count_nonzero(np.any(masks, axis=0))),
    }
    return obj_ids, masks, meta


def load_alltracker_masks(alltracker_dir: Path, frame_id: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    coverage_dir = alltracker_dir / "coverage_masks"
    core = load_gray_bool(coverage_dir / f"frame_{int(frame_id):06d}_core.png", h, w)
    envelope = load_gray_bool(coverage_dir / f"frame_{int(frame_id):06d}_envelope.png", h, w)
    return core, envelope


def component_area_stats(mask: np.ndarray, min_area: int) -> dict[str, Any]:
    if not np.any(mask):
        return {
            "component_count": 0,
            "kept_component_count": 0,
            "max_component_area": 0,
            "area_ge_min_total": 0,
        }
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = [int(stats[idx, cv2.CC_STAT_AREA]) for idx in range(1, int(n_labels))]
    kept = [area for area in areas if area >= int(min_area)]
    return {
        "component_count": int(len(areas)),
        "kept_component_count": int(len(kept)),
        "max_component_area": int(max(areas) if areas else 0),
        "area_ge_min_total": int(sum(kept)),
    }


def component_masks_from_candidate(
    candidate: np.ndarray,
    *,
    min_area: int,
    max_components: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    candidate_bool = candidate.astype(bool)
    h, w = candidate_bool.shape
    if not np.any(candidate_bool):
        return np.zeros((0, h, w), dtype=bool), []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_bool.astype(np.uint8), 8)
    rows: list[tuple[int, int, int, int, int, int]] = []
    for label_id in range(1, int(n_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        rows.append((label_id, area, x, y, bw, bh))
    rows.sort(key=lambda item: item[1], reverse=True)
    if int(max_components) > 0:
        rows = rows[: int(max_components)]
    if not rows:
        return np.zeros((0, h, w), dtype=bool), []
    masks = np.stack([(labels == int(row[0])) for row in rows], axis=0).astype(bool)
    records = [
        {
            "component_rank": int(idx),
            "component_label": int(label_id),
            "component_area": int(area),
            "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
        }
        for idx, (label_id, area, x, y, bw, bh) in enumerate(rows)
    ]
    return masks, records


def _distance_maxima_points(mask: np.ndarray, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    mask_bool = mask.astype(bool)
    count = max(int(count), 0)
    if count <= 0 or not np.any(mask_bool):
        return []
    dist = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(mask_bool)
    chosen: list[tuple[int, int]] = []
    score = dist.copy()
    area = int(np.count_nonzero(mask_bool))
    for _ in range(count):
        if not np.isfinite(score).any() or float(score.max()) <= 0.0:
            idx = int(rng.integers(0, len(ys)))
            y, x = int(ys[idx]), int(xs[idx])
        else:
            y, x = [int(v) for v in np.unravel_index(int(np.argmax(score)), score.shape)]
        chosen.append((y, x))
        radius = max(6, int(round(np.sqrt(float(area) / float(max(count, 1))) * 0.35)))
        cv2.circle(score, (x, y), radius, 0.0, thickness=-1)
    return chosen


def component_prompt_specs_from_candidate(
    candidate: np.ndarray,
    *,
    negative_mask: np.ndarray,
    min_area: int,
    max_components: int,
    positive_points_per_component: int,
    negative_points_per_component: int,
    box_expand_px: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidate_bool = candidate.astype(bool)
    h, w = candidate_bool.shape
    if not np.any(candidate_bool):
        return []
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_bool.astype(np.uint8), 8)
    rows: list[tuple[int, int, int, int, int, int]] = []
    for label_id in range(1, int(n_labels)):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        rows.append((label_id, area, x, y, bw, bh))
    rows.sort(key=lambda item: item[1], reverse=True)
    if int(max_components) > 0:
        rows = rows[: int(max_components)]

    rng = np.random.default_rng(int(seed))
    specs: list[dict[str, Any]] = []
    neg_all = negative_mask.astype(bool)
    expand = max(int(box_expand_px), 0)
    for rank, (label_id, area, x, y, bw, bh) in enumerate(rows):
        comp = labels == int(label_id)
        x0 = max(int(x) - expand, 0)
        y0 = max(int(y) - expand, 0)
        x1 = min(int(x + bw - 1) + expand, w - 1)
        y1 = min(int(y + bh - 1) + expand, h - 1)
        bbox_region = np.zeros((h, w), dtype=bool)
        bbox_region[y0 : y1 + 1, x0 : x1 + 1] = True
        positive_points = _distance_maxima_points(
            comp,
            int(positive_points_per_component),
            rng,
        )
        neg_region = neg_all & bbox_region & ~comp
        negative_points = _distance_maxima_points(
            neg_region,
            int(negative_points_per_component),
            rng,
        )
        specs.append(
            {
                "component_rank": int(rank),
                "component_label": int(label_id),
                "component_area": int(area),
                "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                "positive_points_yx": [[int(py), int(px)] for py, px in positive_points],
                "negative_points_yx": [[int(ny), int(nx)] for ny, nx in negative_points],
                "_component_mask": comp.astype(bool),
            }
        )
    return specs


def run_sam2_box_point_component_repair(
    segmentor: Any,
    rgb: np.ndarray,
    *,
    candidate: np.ndarray,
    negative_mask: np.ndarray,
    min_component_area: int,
    max_components: int,
    positive_points_per_component: int,
    negative_points_per_component: int,
    box_expand_px: int,
    seed: int,
    iou_threshold: float,
    stability_threshold: float,
    stability_score_offset: float,
    model_mask_thresh: float,
    box_nms_thresh: float,
    empty_ratio: float,
    apply_box_nms: bool,
    nms_score_type: str,
    support_min_area: int,
    support_min_ratio: float,
    min_component_completion_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    try:
        from sam2.utils.amg import batched_mask_to_box, calculate_stability_score
    except Exception:
        from efficient_track_anything.utils.amg import batched_mask_to_box, calculate_stability_score
    from torchvision.ops import nms

    h, w = rgb.shape[:2]
    if str(nms_score_type) not in {"pred_iou", "stability"}:
        raise ValueError(f"unsupported nms_score_type={nms_score_type}")
    specs = component_prompt_specs_from_candidate(
        candidate,
        negative_mask=negative_mask,
        min_area=int(min_component_area),
        max_components=int(max_components),
        positive_points_per_component=int(positive_points_per_component),
        negative_points_per_component=int(negative_points_per_component),
        box_expand_px=int(box_expand_px),
        seed=int(seed),
    )

    selected_batches = []
    selected_score_batches = []
    component_records: list[dict[str, Any]] = []
    raw_option_count = 0
    prompt_with_good = 0
    prompt_with_supported = 0
    segmentor.reset_predictor()
    segmentor.set_image(rgb)
    candidate_t = torch.as_tensor(candidate.astype(bool), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for spec in specs:
            pos = [(int(y), int(x), 1) for y, x in spec["positive_points_yx"]]
            neg = [(int(y), int(x), 0) for y, x in spec["negative_points_yx"]]
            point_rows = pos + neg
            coords = None
            labels = None
            if point_rows:
                pts_px = torch.tensor(
                    [[float(x), float(y)] for y, x, _ in point_rows],
                    device="cuda",
                    dtype=torch.float32,
                )
                coords = segmentor._transforms.transform_coords(
                    pts_px.unsqueeze(0),
                    normalize=True,
                    orig_hw=(h, w),
                )
                labels = torch.tensor(
                    [[int(label) for _, _, label in point_rows]],
                    dtype=torch.int,
                    device="cuda",
                )
            box = torch.tensor([spec["bbox_xyxy"]], device="cuda", dtype=torch.float32)
            boxes = segmentor._transforms.transform_boxes(box, normalize=True, orig_hw=(h, w))
            masks, iou_predictions, _ = segmentor._predict(
                coords,
                labels,
                boxes,
                multimask_output=True,
                return_logits=True,
            )
            stability = calculate_stability_score(masks, float(model_mask_thresh), float(stability_score_offset))
            mask_bool = masks > float(model_mask_thresh)
            good = (iou_predictions > float(iou_threshold)) & (stability >= float(stability_threshold))
            raw_option_count += int(good.numel())
            comp_t = torch.as_tensor(spec["_component_mask"].astype(bool), device="cuda")
            areas = mask_bool.sum(dim=(-1, -2), dtype=torch.int64)
            candidate_areas = (mask_bool & candidate_t.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2), dtype=torch.int64)
            component_areas = (mask_bool & comp_t.unsqueeze(0).unsqueeze(0)).sum(dim=(-1, -2), dtype=torch.int64)
            component_completion = component_areas.float() / float(max(int(spec["component_area"]), 1))
            support_ratios = candidate_areas.float() / torch.clamp(areas.float(), min=1.0)
            supported = (
                good
                & (candidate_areas >= int(support_min_area))
                & (support_ratios >= float(support_min_ratio))
                & (component_completion >= float(min_component_completion_ratio))
            )
            prompt_with_good += int(good.any(dim=1).sum().item())
            prompt_with_supported += int(supported.any(dim=1).sum().item())
            selected_idx = None
            if bool(supported.any().item()):
                score = component_areas.clone()
                score[~supported] = -1
                selected_idx = int(score.argmax(dim=1)[0].item())
                selected = mask_bool[0, selected_idx]
                if str(nms_score_type) == "pred_iou":
                    selected_score = iou_predictions[0, selected_idx].float()
                else:
                    selected_score = stability[0, selected_idx].float()
                selected_batches.append(selected.unsqueeze(0))
                selected_score_batches.append(selected_score.reshape(1))
            record = {
                key: value
                for key, value in spec.items()
                if not str(key).startswith("_")
            }
            record.update(
                {
                    "prompt_point_count": int(len(point_rows)),
                    "positive_point_count": int(len(pos)),
                    "negative_point_count": int(len(neg)),
                    "good_option_count": int(good.sum().item()),
                    "supported_option_count": int(supported.sum().item()),
                    "selected_option_index": selected_idx,
                    "best_component_completion": float(component_completion.max().item()) if component_completion.numel() else 0.0,
                    "best_candidate_touch_area": int(candidate_areas.max().item()) if candidate_areas.numel() else 0,
                    "best_support_ratio": float(support_ratios.max().item()) if support_ratios.numel() else 0.0,
                }
            )
            component_records.append(record)

    if selected_batches:
        selected_t = torch.cat(selected_batches, dim=0)
        selected_scores_t = torch.cat(selected_score_batches, dim=0)
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
        "repair_prompt_type": "component_box_plus_positive_negative_points",
        "component_prompt_count": int(len(specs)),
        "raw_multimask_option_count": int(raw_option_count),
        "prompt_with_good_mask_count": int(prompt_with_good),
        "prompt_with_component_supported_mask_count": int(prompt_with_supported),
        "support_min_area": int(support_min_area),
        "support_min_ratio": float(support_min_ratio),
        "min_component_completion_ratio": float(min_component_completion_ratio),
        "positive_points_per_component": int(positive_points_per_component),
        "negative_points_per_component": int(negative_points_per_component),
        "box_expand_px": int(box_expand_px),
        "pre_nms_mask_count": int(pre_nms_count),
        "post_disjoint_mask_count": int(disjoint_np.shape[0]),
        "apply_box_nms": bool(apply_box_nms),
        "nms_score_type": str(nms_score_type),
        "component_records": component_records[:64],
    }
    return disjoint_np, stats


def append_component_fallback_births(
    *,
    filtered_birth_masks: np.ndarray,
    candidate: np.ndarray,
    current_union: np.ndarray,
    core: np.ndarray,
    mode: str,
    min_component_area: int,
    max_components: int,
    min_birth_mask_area: int,
    min_candidate_touch_area: int,
    min_candidate_touch_ratio: float,
    max_existing_overlap_ratio: float,
    max_core_overlap_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode = str(mode)
    if mode not in {"disabled", "when_empty", "always"}:
        raise ValueError(f"unsupported component fallback mode after filter: {mode}")
    base_count = int(filtered_birth_masks.shape[0]) if filtered_birth_masks.size else 0
    fallback_raw, component_records = component_masks_from_candidate(
        candidate,
        min_area=int(min_component_area),
        max_components=int(max_components),
    )
    apply_fallback = bool(mode == "always" or (mode == "when_empty" and base_count == 0))
    fallback_filtered = np.zeros((0, *candidate.shape), dtype=bool)
    fallback_filter_records: list[dict[str, Any]] = []
    if apply_fallback and fallback_raw.size:
        fallback_filtered, fallback_filter_records = filter_birth_masks(
            fallback_raw,
            candidate=candidate,
            current_union=current_union,
            core=core,
            min_birth_mask_area=int(min_birth_mask_area),
            min_candidate_touch_area=int(min_candidate_touch_area),
            min_candidate_touch_ratio=float(min_candidate_touch_ratio),
            max_existing_overlap_ratio=float(max_existing_overlap_ratio),
            max_core_overlap_ratio=float(max_core_overlap_ratio),
        )
    if apply_fallback and fallback_filtered.size:
        if filtered_birth_masks.size:
            combined = np.concatenate([filtered_birth_masks.astype(bool), fallback_filtered.astype(bool)], axis=0)
        else:
            combined = fallback_filtered.astype(bool)
    else:
        combined = filtered_birth_masks.astype(bool) if filtered_birth_masks.size else np.zeros((0, *candidate.shape), dtype=bool)
    record = {
        "mode": mode,
        "applied": bool(apply_fallback),
        "base_filtered_birth_mask_count": int(base_count),
        "candidate_component_raw_count": int(fallback_raw.shape[0]),
        "candidate_component_filtered_count": int(fallback_filtered.shape[0]),
        "output_birth_mask_count": int(combined.shape[0]),
        "min_component_area": int(min_component_area),
        "max_components": int(max_components),
        "component_records": component_records[:64],
        "component_filter_records": fallback_filter_records[:64],
        "audit_note": "candidate component fallback is a bounded repair/diagnostic for point-prompt miss; it is not enabled by default",
    }
    return combined.astype(bool), record


def build_candidate_mask(
    *,
    variant: str,
    uncovered: np.ndarray,
    core: np.ndarray,
    envelope: np.ndarray,
    dilation_px: int,
) -> np.ndarray:
    variant = str(variant)
    if int(dilation_px) > 0:
        kernel = np.ones((2 * int(dilation_px) + 1, 2 * int(dilation_px) + 1), dtype=np.uint8)
        core_for_repel = cv2.dilate(core.astype(np.uint8), kernel, iterations=1) > 0
        envelope_for_gap = cv2.dilate(envelope.astype(np.uint8), kernel, iterations=1) > 0
    else:
        core_for_repel = core.astype(bool)
        envelope_for_gap = envelope.astype(bool)
    if variant == "p3_definite_no_reconcile":
        return (uncovered.astype(bool) & ~envelope_for_gap).astype(bool)
    if variant == "p3_uncertain_band_expansion":
        return (uncovered.astype(bool) & ~core_for_repel).astype(bool)
    if variant == "p4_temporal_persistent_definite":
        return (uncovered.astype(bool) & ~envelope_for_gap).astype(bool)
    if variant == "p4_anchor_period_definite":
        return (uncovered.astype(bool) & ~envelope_for_gap).astype(bool)
    if variant == "p5_protected_core_definite":
        return (uncovered.astype(bool) & ~envelope_for_gap & ~core_for_repel).astype(bool)
    raise ValueError(f"unsupported variant={variant}")


def filter_birth_masks(
    masks: np.ndarray,
    *,
    candidate: np.ndarray,
    current_union: np.ndarray,
    core: np.ndarray,
    min_birth_mask_area: int,
    min_candidate_touch_area: int,
    min_candidate_touch_ratio: float,
    max_existing_overlap_ratio: float,
    max_core_overlap_ratio: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    if masks.size == 0:
        return np.zeros((0, *candidate.shape), dtype=bool), records
    for idx, raw in enumerate(masks.astype(bool)):
        area = int(np.count_nonzero(raw))
        candidate_touch = int(np.count_nonzero(raw & candidate))
        existing_overlap = int(np.count_nonzero(raw & current_union))
        core_overlap = int(np.count_nonzero(raw & core))
        candidate_touch_ratio = float(candidate_touch) / float(max(area, 1))
        existing_overlap_ratio = float(existing_overlap) / float(max(area, 1))
        core_overlap_ratio = float(core_overlap) / float(max(area, 1))
        reasons: list[str] = []
        if area < int(min_birth_mask_area):
            reasons.append("area_lt_min")
        if candidate_touch < int(min_candidate_touch_area):
            reasons.append("candidate_touch_lt_min")
        if candidate_touch_ratio < float(min_candidate_touch_ratio):
            reasons.append("candidate_touch_ratio_lt_min")
        if existing_overlap_ratio > float(max_existing_overlap_ratio):
            reasons.append("existing_overlap_ratio_gt_max")
        if core_overlap_ratio > float(max_core_overlap_ratio):
            reasons.append("core_overlap_ratio_gt_max")
        accepted = not reasons
        if accepted:
            rows.append(raw.astype(bool))
        records.append(
            {
                "raw_index": int(idx),
                "accepted": bool(accepted),
                "reject_reasons": reasons,
                "mask_area": int(area),
                "candidate_touch_area": int(candidate_touch),
                "candidate_touch_ratio": float(candidate_touch_ratio),
                "existing_overlap_ratio": float(existing_overlap_ratio),
                "core_overlap_ratio": float(core_overlap_ratio),
            }
        )
    if rows:
        return np.stack(rows, axis=0).astype(bool), records
    return np.zeros((0, *candidate.shape), dtype=bool), records


def foreground_metrics(pred_label: np.ndarray, ref_label: np.ndarray) -> dict[str, float | int]:
    pred = pred_label > 0
    ref = ref_label > 0
    inter = int(np.count_nonzero(pred & ref))
    pred_area = int(np.count_nonzero(pred))
    ref_area = int(np.count_nonzero(ref))
    union = int(np.count_nonzero(pred | ref))
    return {
        "pred_fg_area": int(pred_area),
        "ref_fg_area": int(ref_area),
        "fg_intersection_area": int(inter),
        "fg_union_area": int(union),
        "fg_iou": float(inter) / float(union) if union else 1.0,
        "fg_precision": float(inter) / float(pred_area) if pred_area else (1.0 if ref_area == 0 else 0.0),
        "fg_recall": float(inter) / float(ref_area) if ref_area else 1.0,
        "false_positive_area": int(np.count_nonzero(pred & ~ref)),
        "false_negative_area": int(np.count_nonzero(ref & ~pred)),
    }


def exact_gap_metrics(pred_label: np.ndarray, x0_label: np.ndarray, x1_label: np.ndarray) -> dict[str, float | int]:
    pred = pred_label > 0
    x0 = x0_label > 0
    x1 = x1_label > 0
    x1_minus_x0 = x1 & ~x0
    x0_minus_x1 = x0 & ~x1
    x1_gap_area = int(np.count_nonzero(x1_minus_x0))
    x0_gap_area = int(np.count_nonzero(x0_minus_x1))
    pred_x1_gap_hit = int(np.count_nonzero(pred & x1_minus_x0))
    pred_x0_gap_hit = int(np.count_nonzero(pred & x0_minus_x1))
    return {
        "x1_minus_x0_area": int(x1_gap_area),
        "x1_minus_x0_pred_recall": float(pred_x1_gap_hit) / float(x1_gap_area) if x1_gap_area else 1.0,
        "x1_minus_x0_pred_hit_area": int(pred_x1_gap_hit),
        "x0_minus_x1_area": int(x0_gap_area),
        "x0_minus_x1_pred_recall": float(pred_x0_gap_hit) / float(x0_gap_area) if x0_gap_area else 1.0,
        "x0_minus_x1_pred_hit_area": int(pred_x0_gap_hit),
        "pred_outside_x1_area": int(np.count_nonzero(pred & ~x1)),
        "pred_outside_x0_area": int(np.count_nonzero(pred & ~x0)),
    }


def decode_frame0_residual_births(
    *,
    cli: argparse.Namespace,
    args: SimpleNamespace,
    segmentor: Any,
    rgb: np.ndarray,
    seed_obj_ids: np.ndarray,
    seed_masks: np.ndarray,
    next_obj_id: int,
    scene_id: str,
    frame_id: int,
    birth_mask_dir: Path,
    h: int,
    w: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]], float]:
    if not bool(cli.enable_frame0_residual_repair):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, h, w), dtype=bool),
            {
                "enabled": False,
                "source": "frame0_seed_only",
                "candidate_area": 0,
                "candidate_component_count": 0,
                "candidate_kept_component_count": 0,
                "candidate_point_count": 0,
                "raw_birth_mask_count": 0,
                "accepted_birth_mask_count": 0,
            },
            [],
            0.0,
        )

    current_union = np.any(seed_masks, axis=0) if seed_masks.size else np.zeros((h, w), dtype=bool)
    candidate_mode = str(cli.frame0_residual_candidate_mode)
    if candidate_mode == "full_union_uncovered":
        protected_union = current_union
        candidate = uncovered_from_masks(seed_masks, h, w).astype(bool)
    elif candidate_mode == "eroded_core_uncovered":
        erosion_px = max(int(cli.frame0_residual_core_erosion_px), 0)
        if erosion_px > 0 and np.any(current_union):
            kernel = np.ones((2 * erosion_px + 1, 2 * erosion_px + 1), dtype=np.uint8)
            protected_union = cv2.erode(current_union.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            protected_union = current_union.copy()
        candidate = ~protected_union
    else:
        raise ValueError(f"unsupported frame0_residual_candidate_mode={candidate_mode}")
    candidate_stats = component_area_stats(candidate, int(cli.frame0_residual_min_component_area))
    point_seed = stable_seed(args.seed, scene_id, frame_id, "phase6-frame0-residual-repair")
    points_yx, point_meta = sample_component_adaptive_points_yx(
        candidate,
        max_points=int(cli.frame0_residual_max_points),
        min_component_area=int(cli.frame0_residual_min_component_area),
        base_points_per_component=int(cli.frame0_residual_base_points_per_component),
        area_per_extra_point=int(cli.frame0_residual_area_per_extra_point),
        max_points_per_component=int(cli.frame0_residual_max_points_per_component),
        seed=int(point_seed),
    )

    birth_t0 = time.time()
    frame0_component_fallback_mode = str(cli.frame0_residual_component_fallback_mode)
    if frame0_component_fallback_mode == "skip_sam":
        raw_birth_masks, component_records = component_masks_from_candidate(
            candidate,
            min_area=int(cli.frame0_residual_component_fallback_min_area),
            max_components=int(cli.frame0_residual_component_fallback_max_components),
        )
        birth_stats = {
            "choice_policy": str(cli.frame0_residual_choice_policy),
            "raw_multimask_option_count": 0,
            "prompt_with_good_mask_count": 0,
            "candidate_supported_option_count": 0,
            "prompt_with_candidate_supported_mask_count": 0,
            "pre_nms_mask_count": 0,
            "post_disjoint_mask_count": int(raw_birth_masks.shape[0]),
            "apply_box_nms": False,
            "nms_score_type": "component_fallback",
            "component_fallback_prompt_decoder_skipped": True,
            "component_fallback_raw_count": int(raw_birth_masks.shape[0]),
            "component_fallback_component_records": component_records[:64],
        }
    elif int(points_yx.shape[0]) > 0:
        raw_birth_masks, birth_stats = run_sam2_point_segment_choice_candidate_support(
            segmentor,
            rgb,
            points_yx=points_yx,
            support_mask=candidate,
            points_per_batch=int(args.points_per_batch),
            choice_policy=str(cli.frame0_residual_choice_policy),
            iou_threshold=float(cli.frame0_residual_pred_iou_thresh),
            stability_threshold=float(cli.frame0_residual_stability_score_thresh),
            stability_score_offset=float(args.stability_score_offset),
            model_mask_thresh=float(args.model_mask_thresh),
            box_nms_thresh=float(args.box_nms_thresh),
            empty_ratio=float(args.empty_ratio),
            apply_box_nms=False,
            nms_score_type="stability",
            support_min_area=int(cli.frame0_residual_min_candidate_touch_area),
            support_min_ratio=float(cli.frame0_residual_min_candidate_touch_ratio),
        )
    else:
        raw_birth_masks = np.zeros((0, h, w), dtype=bool)
        birth_stats = {
            "choice_policy": str(cli.frame0_residual_choice_policy),
            "raw_multimask_option_count": 0,
            "prompt_with_good_mask_count": 0,
            "candidate_supported_option_count": 0,
            "prompt_with_candidate_supported_mask_count": 0,
            "pre_nms_mask_count": 0,
            "post_disjoint_mask_count": 0,
            "apply_box_nms": False,
            "nms_score_type": "stability",
        }
    decode_sec = time.time() - birth_t0

    filtered_birth_masks, filter_records = filter_birth_masks(
        raw_birth_masks,
        candidate=candidate,
        current_union=protected_union,
        core=protected_union,
        min_birth_mask_area=int(cli.frame0_residual_min_birth_mask_area),
        min_candidate_touch_area=int(cli.frame0_residual_min_candidate_touch_area),
        min_candidate_touch_ratio=float(cli.frame0_residual_min_candidate_touch_ratio),
        max_existing_overlap_ratio=float(cli.frame0_residual_max_existing_overlap_ratio),
        max_core_overlap_ratio=float(cli.frame0_residual_max_core_overlap_ratio),
    )
    if frame0_component_fallback_mode in {"when_empty", "always"}:
        filtered_birth_masks, frame0_component_fallback_record = append_component_fallback_births(
            filtered_birth_masks=filtered_birth_masks,
            candidate=candidate,
            current_union=protected_union,
            core=protected_union,
            mode=("when_empty" if frame0_component_fallback_mode == "when_empty" else "always"),
            min_component_area=int(cli.frame0_residual_component_fallback_min_area),
            max_components=int(cli.frame0_residual_component_fallback_max_components),
            min_birth_mask_area=int(cli.frame0_residual_min_birth_mask_area),
            min_candidate_touch_area=int(cli.frame0_residual_min_candidate_touch_area),
            min_candidate_touch_ratio=float(cli.frame0_residual_min_candidate_touch_ratio),
            max_existing_overlap_ratio=float(cli.frame0_residual_max_existing_overlap_ratio),
            max_core_overlap_ratio=float(cli.frame0_residual_max_core_overlap_ratio),
        )
    else:
        frame0_component_fallback_record = {
            "mode": frame0_component_fallback_mode,
            "applied": bool(frame0_component_fallback_mode == "skip_sam"),
            "base_filtered_birth_mask_count": int(filtered_birth_masks.shape[0]) if filtered_birth_masks.size else 0,
            "candidate_component_raw_count": int(raw_birth_masks.shape[0]) if frame0_component_fallback_mode == "skip_sam" else 0,
            "candidate_component_filtered_count": int(filtered_birth_masks.shape[0]) if frame0_component_fallback_mode == "skip_sam" else 0,
            "output_birth_mask_count": int(filtered_birth_masks.shape[0]) if filtered_birth_masks.size else 0,
            "audit_note": "component fallback disabled or prompt decoder skipped before standard filtering",
        }
    if filtered_birth_masks.size and int(cli.frame0_residual_max_births) > 0:
        areas = np.count_nonzero(filtered_birth_masks.reshape(filtered_birth_masks.shape[0], -1), axis=1)
        keep_idx = np.argsort(areas)[::-1][: int(cli.frame0_residual_max_births)]
        keep_idx.sort()
        accepted_birth_masks_raw = filtered_birth_masks[keep_idx]
    else:
        accepted_birth_masks_raw = filtered_birth_masks

    accepted_masks: list[np.ndarray] = []
    accepted_ids: list[int] = []
    repair_birth_records: list[dict[str, Any]] = []
    write_union = protected_union.copy()
    for local_idx, raw_mask in enumerate(accepted_birth_masks_raw.astype(bool)):
        write_mask = raw_mask & ~write_union
        if int(np.count_nonzero(write_mask)) < int(cli.frame0_residual_min_birth_mask_area):
            continue
        obj_id = int(next_obj_id + len(accepted_ids))
        write_union[write_mask] = True
        accepted_ids.append(obj_id)
        accepted_masks.append(write_mask.astype(bool))
        mask_path = birth_mask_dir / f"frame_{frame_id:06d}_obj_{obj_id:06d}_frame0_residual_repair.png"
        cv2.imwrite(str(mask_path), write_mask.astype(np.uint8) * 255)
        repair_birth_records.append(
            {
                "scene_id": str(scene_id),
                "chunk_frame_index": 0,
                "frame_id": int(frame_id),
                "obj_id": int(obj_id),
                "local_birth_index": int(local_idx),
                "source": "frame0_residual_repair",
                "mask_path": str(mask_path),
                "raw_mask_area": int(np.count_nonzero(raw_mask)),
                "mask_area": int(np.count_nonzero(write_mask)),
                "candidate_touch_area": int(np.count_nonzero(raw_mask & candidate)),
                "existing_overlap_area": int(np.count_nonzero(raw_mask & current_union)),
                "protected_core_overlap_area": int(np.count_nonzero(raw_mask & protected_union)),
            }
        )

    accepted_ids_np = np.asarray(accepted_ids, dtype=np.int64)
    accepted_masks_np = (
        np.stack(accepted_masks, axis=0).astype(bool)
        if accepted_masks
        else np.zeros((0, h, w), dtype=bool)
    )
    record = {
        "enabled": True,
        "source": "frame0_seed_plus_residual_repair",
        "candidate_mode": candidate_mode,
        "protected_union_area": int(np.count_nonzero(protected_union)),
        "current_union_area": int(np.count_nonzero(current_union)),
        "core_erosion_px": int(cli.frame0_residual_core_erosion_px),
        "candidate_area": int(np.count_nonzero(candidate)),
        "candidate_component_count": int(candidate_stats["component_count"]),
        "candidate_kept_component_count": int(candidate_stats["kept_component_count"]),
        "candidate_max_component_area": int(candidate_stats["max_component_area"]),
        "candidate_area_ge_min_total": int(candidate_stats["area_ge_min_total"]),
        "candidate_point_count": int(points_yx.shape[0]),
        "candidate_sampling_meta": point_meta,
        "birth_decode_runtime_sec": float(decode_sec),
        "raw_birth_mask_count": int(raw_birth_masks.shape[0]),
        "filtered_birth_mask_count": int(filtered_birth_masks.shape[0]),
        "accepted_birth_mask_count": int(accepted_ids_np.size),
        "birth_stats": birth_stats,
        "birth_filter_records": filter_records[:64],
        "component_fallback": frame0_component_fallback_record,
        "accepted_birth_write_policy": (
            "write_only_base_uncovered_pixels"
            if candidate_mode == "full_union_uncovered"
            else "write_only_eroded_core_uncovered_pixels"
        ),
    }
    return accepted_ids_np, accepted_masks_np, record, repair_birth_records, float(decode_sec)


def run(cli: argparse.Namespace) -> None:
    import torch

    config_path = resolve_path(cli.config)
    args = make_baseline_args(config_path, cli)
    args.scene_id = str(cli.scene_id or args.scene_id)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    rgb_root = resolve_path(args.rgb_root) / args.scene_id / "color"
    frame_paths = [rgb_root / f"{frame_id}.jpg" for frame_id in frame_ids]
    missing = [str(path) for path in frame_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[:5])

    output_root = resolve_path(cli.output_root)
    label_dir = output_root / "labels"
    overlay_dir = output_root / "overlays"
    sheet_dir = output_root / "sheets"
    video_dir_out = output_root / "videos"
    birth_mask_dir = output_root / "birth_masks"
    output_dirs = [label_dir, birth_mask_dir]
    if not bool(cli.skip_chunk_overlays):
        output_dirs.append(overlay_dir)
    if not bool(cli.skip_chunk_sheets):
        output_dirs.append(sheet_dir)
    if not bool(cli.skip_chunk_video):
        output_dirs.append(video_dir_out)
    for directory in output_dirs:
        directory.mkdir(parents=True, exist_ok=True)

    rgbs = [read_rgb(path) for path in frame_paths]
    h, w = rgbs[0].shape[:2]
    video_dir = make_numeric_frame_dir(frame_paths, output_root)
    suffix_video_dirs: dict[int, Path] = {}

    def propagation_video_scope(seed_idx: int, stop_exclusive: int) -> tuple[Path, int, int, int, int | None, bool, str]:
        seed_idx = int(seed_idx)
        stop_exclusive = int(stop_exclusive)
        scope = str(cli.propagation_video_scope)
        if scope == "suffix_absolute":
            if not bool(cli.use_video_feature_bank):
                raise ValueError("--propagation-video-scope suffix_absolute requires --use-video-feature-bank")
            if seed_idx > 0:
                if seed_idx not in suffix_video_dirs:
                    suffix_video_dirs[seed_idx] = make_numeric_frame_suffix_dir(frame_paths, output_root, seed_idx)
                return (
                    suffix_video_dirs[seed_idx],
                    seed_idx,
                    stop_exclusive,
                    0,
                    stop_exclusive,
                    True,
                    "suffix_absolute",
                )
            return video_dir, seed_idx, stop_exclusive, 0, None, False, "full"
        if str(cli.propagation_video_scope) == "suffix" and seed_idx > 0:
            if seed_idx not in suffix_video_dirs:
                suffix_video_dirs[seed_idx] = make_numeric_frame_suffix_dir(frame_paths, output_root, seed_idx)
            return (
                suffix_video_dirs[seed_idx],
                0,
                max(0, stop_exclusive - seed_idx),
                seed_idx,
                None,
                False,
                "suffix",
            )
        return video_dir, seed_idx, stop_exclusive, 0, None, False, "full"

    x0_label_paths, x0_summary = load_label_paths(resolve_path(cli.x0_summary))
    x1_label_paths, x1_summary = load_label_paths(resolve_path(cli.x1_summary))
    alltracker_dir = resolve_path(cli.alltracker_dir)
    alltracker_summary_path = alltracker_dir / "alltracker_contract_summary.json"
    alltracker_summary = json.loads(alltracker_summary_path.read_text(encoding="utf-8"))
    seed_obj_ids, seed_masks, seed_meta = load_frame0_seed_rows(
        resolve_path(cli.frame0_birth_records),
        frame_ids,
        args.scene_id,
        h,
        w,
    )
    next_obj_id = int(seed_obj_ids.max()) + 1

    t_setup = time.time()
    models = setup_models(args)
    setup_sec = time.time() - t_setup
    segmentor = models["segmentor"]
    tracker_model = models["tracker_model"]
    tracker_num_maskmem_original = getattr(tracker_model, "num_maskmem", None)
    if int(cli.tracker_num_maskmem_override) > 0:
        setattr(tracker_model, "num_maskmem", int(cli.tracker_num_maskmem_override))
    tracker_max_obj_ptrs_in_encoder_original = getattr(tracker_model, "max_obj_ptrs_in_encoder", None)
    if int(cli.tracker_max_obj_ptrs_in_encoder_override) > 0:
        setattr(tracker_model, "max_obj_ptrs_in_encoder", int(cli.tracker_max_obj_ptrs_in_encoder_override))
    stream_max_cond_frames_in_attn_original = getattr(tracker_model, "max_cond_frames_in_attn", None)
    if int(cli.stream_max_cond_frames_in_attn) >= 0:
        setattr(tracker_model, "max_cond_frames_in_attn", int(cli.stream_max_cond_frames_in_attn))
    stream_clear_noncond_mem_around_input_original = getattr(tracker_model, "clear_non_cond_mem_around_input", None)
    stream_clear_noncond_mem_for_multi_obj_original = getattr(tracker_model, "clear_non_cond_mem_for_multi_obj", None)
    if bool(cli.stream_clear_noncond_mem_around_input):
        setattr(tracker_model, "clear_non_cond_mem_around_input", True)
        setattr(tracker_model, "clear_non_cond_mem_for_multi_obj", True)

    feature_bank_info: dict[str, Any] = empty_feature_bank_summary()
    total_t0 = time.time()
    if bool(cli.use_video_feature_bank):
        feature_bank_info = install_video_feature_bank_patch(
            tracker_model,
            frame_ids=frame_ids,
            frame_paths=frame_paths,
            storage_device=str(cli.video_feature_bank_storage_device),
            video_gpu_hot_window=int(cli.video_gpu_hot_window),
        )

    frame0_residual_ids, frame0_residual_masks, frame0_residual_record, frame0_birth_records, frame0_birth_decode_sec = (
        decode_frame0_residual_births(
            cli=cli,
            args=args,
            segmentor=segmentor,
            rgb=rgbs[0],
            seed_obj_ids=seed_obj_ids,
            seed_masks=seed_masks,
            next_obj_id=next_obj_id,
            scene_id=args.scene_id,
            frame_id=int(frame_ids[0]),
            birth_mask_dir=birth_mask_dir,
            h=h,
            w=w,
        )
    )
    next_obj_id += int(frame0_residual_ids.size)

    acc_ids: list[list[int]] = [[] for _ in frame_ids]
    acc_masks: list[list[np.ndarray]] = [[] for _ in frame_ids]
    to_prop_ids: list[np.ndarray] = [np.zeros((0,), dtype=np.int64) for _ in frame_ids]
    to_prop_masks: list[np.ndarray] = [np.zeros((0, h, w), dtype=bool) for _ in frame_ids]
    to_prop_ids[0] = (
        np.concatenate([seed_obj_ids, frame0_residual_ids]).astype(np.int64)
        if frame0_residual_ids.size
        else seed_obj_ids
    )
    to_prop_masks[0] = (
        np.concatenate([seed_masks, frame0_residual_masks], axis=0).astype(bool)
        if frame0_residual_masks.size
        else seed_masks
    )

    overlay_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    birth_records: list[dict[str, Any]] = list(frame0_birth_records)
    propagation_records: list[dict[str, Any]] = []
    total_tracking_sec = 0.0
    total_birth_decode_sec = float(frame0_birth_decode_sec)
    total_frame_output_sec = 0.0
    total_label_build_sec = 0.0
    total_label_write_sec = 0.0
    total_diagnostic_sec = 0.0
    total_overlay_write_sec = 0.0
    total_birth_mask_write_sec = 0.0
    video_state_template: dict[str, Any] | None = None
    video_state_template_init_sec = 0.0
    if bool(cli.reuse_video_state_template):
        if str(cli.propagation_video_scope) != "full":
            raise ValueError("--reuse-video-state-template currently requires --propagation-video-scope full")
        template_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            video_state_template = tracker_model.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=False,
            )
        video_state_template["cached_features"] = {}
        video_state_template_init_sec = time.time() - template_t0
        total_tracking_sec += float(video_state_template_init_sec)

    def write_frame_output(
        chunk_idx: int,
        current_ids: np.ndarray,
        current_masks: np.ndarray,
        birth_ids: np.ndarray,
        birth_masks: np.ndarray,
        frame_record_extra: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal total_frame_output_sec
        nonlocal total_label_build_sec
        nonlocal total_label_write_sec
        nonlocal total_diagnostic_sec
        nonlocal total_overlay_write_sec
        frame_output_t0 = time.time()
        label_build_t0 = time.time()
        if current_masks.size and birth_masks.size:
            rows_pre = np.concatenate([current_masks.astype(bool), birth_masks.astype(bool)], axis=0)
            ids_pre = np.concatenate([current_ids.astype(np.int64), birth_ids.astype(np.int64)], axis=0)
        elif current_masks.size:
            rows_pre = current_masks.astype(bool)
            ids_pre = current_ids.astype(np.int64)
        elif birth_masks.size:
            rows_pre = birth_masks.astype(bool)
            ids_pre = birth_ids.astype(np.int64)
        else:
            rows_pre = np.zeros((0, h, w), dtype=bool)
            ids_pre = np.zeros((0,), dtype=np.int64)
        if rows_pre.size:
            rows_disjoint, keep = disjoin_keep_order(rows_pre, h, w, empty_ratio=float(args.empty_ratio))
            masks_out = rows_disjoint[keep]
            ids_out = ids_pre[keep]
        else:
            masks_out = np.zeros((0, h, w), dtype=bool)
            ids_out = np.zeros((0,), dtype=np.int64)
        label = label_from_id_masks(ids_out, masks_out, h, w)
        total_label_build_sec += float(time.time() - label_build_t0)
        frame_id = int(frame_ids[chunk_idx])
        label_path = label_dir / f"frame_{frame_id:06d}.png"
        label_write_t0 = time.time()
        cv2.imwrite(str(label_path), label)
        total_label_write_sec += float(time.time() - label_write_t0)

        diagnostic_t0 = time.time()
        x0_label_path = x0_label_paths.get(frame_id)
        if x0_label_path is None and not bool(cli.allow_missing_x0_diagnostics):
            raise KeyError(
                {
                    "missing_x0_diagnostic_label_for_frame": int(frame_id),
                    "x0_summary": str(resolve_path(cli.x0_summary)),
                    "repair": "provide an X0 summary covering every requested frame or rerun with --allow-missing-x0-diagnostics for visualization-only/full-scene exploration",
                }
            )
        x0_label = load_label(x0_label_path, h, w) if x0_label_path is not None else None
        x1_label_path = x1_label_paths.get(frame_id)
        if x1_label_path is None and not bool(cli.allow_missing_x1_diagnostics):
            raise KeyError(
                {
                    "missing_x1_diagnostic_label_for_frame": int(frame_id),
                    "x1_summary": str(resolve_path(cli.x1_summary)),
                    "repair": "provide an X1 summary covering every requested frame or rerun with --allow-missing-x1-diagnostics for visualization-only/full-scene exploration",
                }
            )
        x1_label = load_label(x1_label_path, h, w) if x1_label_path is not None else None
        x0_metrics = foreground_metrics(label, x0_label) if x0_label is not None else None
        x1_metrics = foreground_metrics(label, x1_label) if x1_label is not None else None
        gap_metrics = (
            exact_gap_metrics(label, x0_label, x1_label)
            if x0_label is not None and x1_label is not None
            else {
                "x0_diagnostic_available": False,
                "x1_diagnostic_available": bool(x1_label is not None),
                "x0_missing_frame_id": int(frame_id),
                "x0_missing_reason": (
                    "x0_summary_does_not_cover_this_frame" if x0_label is None else ""
                ),
                "x1_missing_reason": (
                    "x1_summary_does_not_cover_this_frame" if x1_label is None else ""
                ),
                "exact_x0_x1_gap_metrics_available": False,
            }
        )
        total_diagnostic_sec += float(time.time() - diagnostic_t0)
        stats = mask_stats(label)
        overlay_path: Path | None = None
        if not bool(cli.skip_chunk_overlays):
            overlay_t0 = time.time()
            overlay = overlay_label(rgbs[chunk_idx], label)
            x0_iou_text = f"{float(x0_metrics['fg_iou']):.4f}" if x0_metrics is not None else "n/a"
            x1_iou_text = f"{float(x1_metrics['fg_iou']):.4f}" if x1_metrics is not None else "n/a"
            annotated = annotate_frame(
                overlay,
                f"phase6 {cli.variant} frame {chunk_idx:02d} / id {frame_id}",
                [
                    f"ids={stats['visible_id_count']} births={int(birth_ids.size)} cand={int(frame_record_extra.get('candidate_area', 0))}",
                    f"x1_iou={x1_iou_text} x0_iou={x0_iou_text}",
                ],
            )
            overlay_path = overlay_dir / f"frame_{chunk_idx:02d}_id_{frame_id:06d}.jpg"
            annotated.save(overlay_path, quality=95)
            total_overlay_write_sec += float(time.time() - overlay_t0)
            overlay_paths.append(overlay_path)
        record = {
            "chunk_frame_index": int(chunk_idx),
            "frame_id": int(frame_id),
            "label_path": str(label_path),
            "overlay_path": str(overlay_path) if overlay_path is not None else None,
            "pre_disjoin_mask_count": int(rows_pre.shape[0]),
            "output_object_id_count": int(ids_out.size),
            "visible_id_count": int(stats["visible_id_count"]),
            "foreground_ratio": float(stats["foreground_ratio"]),
            "birth_count": int(birth_ids.size),
            "x0_foreground_metrics": x0_metrics,
            "x0_diagnostic_available": bool(x0_metrics is not None),
            "x1_foreground_metrics": x1_metrics,
            "x1_diagnostic_available": bool(x1_metrics is not None),
            "exact_gap_diagnostic_metrics": gap_metrics,
        }
        record.update(frame_record_extra)
        records.append(record)
        total_frame_output_sec += float(time.time() - frame_output_t0)
        return ids_out.astype(np.int64, copy=False), masks_out.astype(bool, copy=False)

    rolling_prev_ids, rolling_prev_masks = write_frame_output(
        0,
        seed_obj_ids,
        seed_masks,
        frame0_residual_ids,
        frame0_residual_masks,
        frame0_residual_record,
    )

    scheduler_mode = str(cli.scheduler_mode)
    stream_state: dict[str, Any] | None = None
    if scheduler_mode == "streaming_state":
        stream_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            stream_state = tracker_model.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                async_loading_frames=False,
            )
        stream_init_sec = time.time() - stream_t0
        total_tracking_sec += stream_init_sec
        stream_add_t0 = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            add_masks_to_stream_state(
                tracker_model,
                stream_state,
                tracker=str(args.tracker_backend),
                frame_idx=0,
                obj_ids=to_prop_ids[0],
                masks=to_prop_masks[0],
            )
            frame0_repairs = reconcile_stream_state_object_count(
                tracker_model,
                stream_state,
                repair_mode=str(cli.stream_state_repair_mode),
            )
        frame0_add_sec = time.time() - stream_add_t0
        total_tracking_sec += frame0_add_sec
        propagation_records.append(
            {
                "scheduler_mode": "streaming_state",
                "stream_state_init_runtime_sec": float(stream_init_sec),
                "stream_state_frame0_add_runtime_sec": float(frame0_add_sec),
                "stream_state_frame0_repair_count": int(len(frame0_repairs)),
                "stream_state_frame0_repairs": frame0_repairs[:16],
                "seed_chunk_frame_index": 0,
                "seed_frame_id": int(frame_ids[0]),
                "seed_object_count": int(to_prop_ids[0].size),
            }
        )
    elif scheduler_mode not in {"independent_anchor", "rolling_one_step"}:
        raise ValueError(f"unsupported scheduler_mode={scheduler_mode}")

    persistent_candidate_hits = np.zeros((h, w), dtype=np.uint8)
    birth_anchor_forced_event_count = 0
    for prev_idx in range(len(frame_ids) - 1):
        chunk_idx = prev_idx + 1
        frame_id = int(frame_ids[chunk_idx])
        if stream_state is not None:
            stream_prune_before_infer = prune_stream_noncond_memory(
                stream_state,
                current_frame_idx=int(chunk_idx),
                keep_window=int(cli.stream_prune_noncond_keep_window),
            )
            prop_t0 = time.time()
            current_ids_pre, current_masks_pre = infer_stream_frame(
                tracker_model,
                stream_state,
                frame_idx=int(chunk_idx),
            )
            prop_sec = time.time() - prop_t0
            total_tracking_sec += prop_sec
            stream_prune_after_infer = prune_stream_noncond_memory(
                stream_state,
                current_frame_idx=int(chunk_idx),
                keep_window=int(cli.stream_prune_noncond_keep_window),
            )
            if current_masks_pre.size:
                current_masks_all, keep = disjoin_keep_order(
                    current_masks_pre,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                )
                current_masks = current_masks_all[keep]
                current_ids = current_ids_pre[keep]
            else:
                current_ids = np.zeros((0,), dtype=np.int64)
                current_masks = np.zeros((0, h, w), dtype=bool)
            propagation_records.append(
                {
                    "scheduler_mode": "streaming_state",
                    "seed_chunk_frame_index": int(prev_idx),
                    "seed_frame_id": int(frame_ids[prev_idx]),
                    "target_chunk_frame_index": int(chunk_idx),
                    "target_frame_id": int(frame_id),
                    "propagation_runtime_sec": float(prop_sec),
                    "stream_infer_pre_disjoin_count": int(current_masks_pre.shape[0]),
                    "stream_infer_post_disjoin_count": int(current_masks.shape[0]),
                    "stream_prune_before_infer": stream_prune_before_infer,
                    "stream_prune_after_infer": stream_prune_after_infer,
                }
            )
        elif scheduler_mode == "rolling_one_step":
            obj_ids = rolling_prev_ids
            masks = rolling_prev_masks
            prop_t0 = time.time()
            propagated = {}
            if masks.size:
                (
                    prop_video_dir,
                    prop_seed_frame,
                    prop_total_frames,
                    prop_frame_offset,
                    prop_state_num_frames,
                    prop_clear_cache,
                    prop_scope,
                ) = propagation_video_scope(
                    prev_idx,
                    min(len(frame_ids), int(chunk_idx) + 1),
                )
                propagated = propagate_new_masks_chunked(
                    tracker_model,
                    tracker=str(args.tracker_backend),
                    video_dir=prop_video_dir,
                    seed_frame=int(prop_seed_frame),
                    obj_ids=obj_ids,
                    masks=masks,
                    total_frames=int(prop_total_frames),
                    offload_video_to_cpu=bool(args.offload_video_to_cpu),
                    offload_state_to_cpu=bool(args.offload_state_to_cpu),
                    chunk_size=int(args.propagation_chunk_size),
                    feature_bank_frame_offset=int(prop_frame_offset),
                    state_num_frames_override=prop_state_num_frames,
                    clear_cached_features_after_init=bool(prop_clear_cache),
                    video_state_template=video_state_template if str(prop_scope) == "full" else None,
                )
            else:
                prop_scope = str(cli.propagation_video_scope)
                prop_frame_offset = 0
                prop_state_num_frames = None
                prop_clear_cache = False
            prop_sec = time.time() - prop_t0
            total_tracking_sec += prop_sec
            frame_outputs = propagated.get(int(chunk_idx - prop_frame_offset), {})
            if frame_outputs:
                current_ids_pre = np.asarray([int(v) for v in frame_outputs.keys()], dtype=np.int64)
                current_masks_pre = np.stack(
                    [frame_outputs[int(obj_id)].astype(bool) for obj_id in current_ids_pre.tolist()],
                    axis=0,
                )
                current_masks_all, keep = disjoin_keep_order(
                    current_masks_pre,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                )
                current_masks = current_masks_all[keep]
                current_ids = current_ids_pre[keep]
            else:
                current_ids = np.zeros((0,), dtype=np.int64)
                current_masks = np.zeros((0, h, w), dtype=bool)
            propagation_records.append(
                {
                    "scheduler_mode": "rolling_one_step",
                    "seed_chunk_frame_index": int(prev_idx),
                    "seed_frame_id": int(frame_ids[prev_idx]),
                    "target_chunk_frame_index": int(chunk_idx),
                    "target_frame_id": int(frame_id),
                    "seed_object_count": int(obj_ids.size),
                    "propagation_runtime_sec": float(prop_sec),
                    "propagation_video_scope": str(prop_scope),
                    "propagation_frame_offset": int(prop_frame_offset),
                    "propagation_state_num_frames_override": (
                        int(prop_state_num_frames) if prop_state_num_frames is not None else None
                    ),
                    "propagation_clear_cached_features_after_init": bool(prop_clear_cache),
                    "propagation_reuse_video_state_template": bool(video_state_template is not None and str(prop_scope) == "full"),
                    "target_output_mask_count": int(current_ids.size),
                }
            )
        else:
            obj_ids = to_prop_ids[prev_idx]
            masks = to_prop_masks[prev_idx]
            prop_t0 = time.time()
            propagated = {}
            hybrid_propagated_parts: list[dict[int, dict[int, np.ndarray]]] = []
            frame0_hybrid_record: dict[str, Any] = {}
            propagation_chunk_runtime_records: list[dict[str, Any]] = []
            if masks.size:
                (
                    prop_video_dir,
                    prop_seed_frame,
                    prop_total_frames,
                    prop_frame_offset,
                    prop_state_num_frames,
                    prop_clear_cache,
                    prop_scope,
                ) = propagation_video_scope(
                    prev_idx,
                    len(frame_ids),
                )
                if int(prev_idx) == 0:
                    frame0_hybrid_record = {
                        "frame0_hybrid_enabled": False,
                        "frame0_hybrid_small_num_maskmem": (
                            int(cli.frame0_hybrid_small_num_maskmem)
                            if int(cli.frame0_hybrid_small_num_maskmem) > 0
                            else None
                        ),
                        "frame0_hybrid_full_topk": (
                            int(cli.frame0_hybrid_full_topk) if int(cli.frame0_hybrid_full_topk) > 0 else None
                        ),
                        "frame0_hybrid_full_min_area": (
                            int(cli.frame0_hybrid_full_min_area)
                            if int(cli.frame0_hybrid_full_min_area) > 0
                            else None
                        ),
                    }
                frame0_hybrid_requested = (
                    int(prev_idx) == 0
                    and int(cli.frame0_hybrid_small_num_maskmem) > 0
                    and (
                        int(cli.frame0_hybrid_full_topk) > 0
                        or int(cli.frame0_hybrid_full_min_area) > 0
                    )
                )
                if frame0_hybrid_requested:
                    split = split_frame0_hybrid_groups(
                        obj_ids,
                        masks,
                        full_topk=int(cli.frame0_hybrid_full_topk),
                        full_min_area=int(cli.frame0_hybrid_full_min_area),
                    )
                    full_indices = split["full_indices"]
                    small_indices = split["small_indices"]
                    frame0_hybrid_record.update(
                        {
                            "frame0_hybrid_full_count": int(full_indices.size),
                            "frame0_hybrid_small_count": int(small_indices.size),
                            "frame0_hybrid_full_area_sum": int(split["full_area_sum"]),
                            "frame0_hybrid_small_area_sum": int(split["small_area_sum"]),
                            "frame0_hybrid_full_ids": split["full_ids"],
                            "frame0_hybrid_small_ids": split["small_ids"],
                            "frame0_hybrid_full_num_maskmem": (
                                int(getattr(tracker_model, "num_maskmem"))
                                if getattr(tracker_model, "num_maskmem", None) is not None
                                else None
                            ),
                            "frame0_hybrid_merge_order": "original_obj_id_order",
                        }
                    )
                    if int(full_indices.size) > 0 and int(small_indices.size) > 0:
                        full_t0 = time.time()
                        full_propagated = propagate_new_masks_chunked(
                            tracker_model,
                            tracker=str(args.tracker_backend),
                            video_dir=prop_video_dir,
                            seed_frame=int(prop_seed_frame),
                            obj_ids=obj_ids[full_indices],
                            masks=masks[full_indices],
                            total_frames=int(prop_total_frames),
                            offload_video_to_cpu=bool(args.offload_video_to_cpu),
                            offload_state_to_cpu=bool(args.offload_state_to_cpu),
                            chunk_size=int(args.propagation_chunk_size),
                            feature_bank_frame_offset=int(prop_frame_offset),
                            state_num_frames_override=prop_state_num_frames,
                            clear_cached_features_after_init=bool(prop_clear_cache),
                            video_state_template=video_state_template if str(prop_scope) == "full" else None,
                            chunk_runtime_records=propagation_chunk_runtime_records,
                        )
                        full_sec = time.time() - full_t0
                        old_num_maskmem = getattr(tracker_model, "num_maskmem", None)
                        small_t0 = time.time()
                        try:
                            setattr(tracker_model, "num_maskmem", int(cli.frame0_hybrid_small_num_maskmem))
                            small_propagated = propagate_new_masks_chunked(
                                tracker_model,
                                tracker=str(args.tracker_backend),
                                video_dir=prop_video_dir,
                                seed_frame=int(prop_seed_frame),
                                obj_ids=obj_ids[small_indices],
                                masks=masks[small_indices],
                                total_frames=int(prop_total_frames),
                                offload_video_to_cpu=bool(args.offload_video_to_cpu),
                                offload_state_to_cpu=bool(args.offload_state_to_cpu),
                                chunk_size=int(args.propagation_chunk_size),
                                feature_bank_frame_offset=int(prop_frame_offset),
                                state_num_frames_override=prop_state_num_frames,
                                clear_cached_features_after_init=bool(prop_clear_cache),
                                video_state_template=video_state_template if str(prop_scope) == "full" else None,
                                chunk_runtime_records=propagation_chunk_runtime_records,
                            )
                        finally:
                            if old_num_maskmem is not None:
                                setattr(tracker_model, "num_maskmem", old_num_maskmem)
                        small_sec = time.time() - small_t0
                        hybrid_propagated_parts = [full_propagated, small_propagated]
                        frame0_hybrid_record.update(
                            {
                                "frame0_hybrid_enabled": True,
                                "frame0_hybrid_full_runtime_sec": float(full_sec),
                                "frame0_hybrid_small_runtime_sec": float(small_sec),
                                "frame0_hybrid_small_num_maskmem_effective": int(cli.frame0_hybrid_small_num_maskmem),
                                "frame0_hybrid_serial_runtime_note": (
                                    "full and small cohorts are run serially in this probe; "
                                    "the record is for visual feasibility and cost decomposition"
                                ),
                            }
                        )
                    else:
                        frame0_hybrid_record["frame0_hybrid_disabled_reason"] = "empty_full_or_small_group"
                if not hybrid_propagated_parts:
                    propagated = propagate_new_masks_chunked(
                        tracker_model,
                        tracker=str(args.tracker_backend),
                        video_dir=prop_video_dir,
                        seed_frame=int(prop_seed_frame),
                        obj_ids=obj_ids,
                        masks=masks,
                        total_frames=int(prop_total_frames),
                        offload_video_to_cpu=bool(args.offload_video_to_cpu),
                        offload_state_to_cpu=bool(args.offload_state_to_cpu),
                        chunk_size=int(args.propagation_chunk_size),
                        feature_bank_frame_offset=int(prop_frame_offset),
                        state_num_frames_override=prop_state_num_frames,
                        clear_cached_features_after_init=bool(prop_clear_cache),
                        video_state_template=video_state_template if str(prop_scope) == "full" else None,
                        chunk_runtime_records=propagation_chunk_runtime_records,
                    )
            else:
                prop_scope = str(cli.propagation_video_scope)
                prop_frame_offset = 0
                prop_state_num_frames = None
                prop_clear_cache = False
            prop_sec = time.time() - prop_t0
            total_tracking_sec += prop_sec
            future_output_count = 0
            if hybrid_propagated_parts:
                future_idx_locals = sorted(
                    {
                        int(future_idx_local)
                        for part in hybrid_propagated_parts
                        for future_idx_local in part.keys()
                    }
                )
                original_obj_order = [int(v) for v in obj_ids.tolist()]
                for future_idx_local in future_idx_locals:
                    future_idx = int(future_idx_local) + int(prop_frame_offset)
                    if int(future_idx) <= int(prev_idx) or int(future_idx) >= len(frame_ids):
                        continue
                    frame_outputs_by_id: dict[int, np.ndarray] = {}
                    for part in hybrid_propagated_parts:
                        for obj_id, mask in part.get(int(future_idx_local), {}).items():
                            frame_outputs_by_id[int(obj_id)] = mask
                    for obj_id in original_obj_order:
                        if int(obj_id) not in frame_outputs_by_id:
                            continue
                        acc_ids[int(future_idx)].append(int(obj_id))
                        acc_masks[int(future_idx)].append(frame_outputs_by_id[int(obj_id)].astype(bool))
                        future_output_count += 1
            else:
                for future_idx_local, frame_outputs in propagated.items():
                    future_idx = int(future_idx_local) + int(prop_frame_offset)
                    if int(future_idx) <= int(prev_idx) or int(future_idx) >= len(frame_ids):
                        continue
                    for obj_id, mask in frame_outputs.items():
                        acc_ids[int(future_idx)].append(int(obj_id))
                        acc_masks[int(future_idx)].append(mask.astype(bool))
                        future_output_count += 1
            prop_record = {
                "scheduler_mode": "independent_anchor",
                "seed_chunk_frame_index": int(prev_idx),
                "seed_frame_id": int(frame_ids[prev_idx]),
                "seed_object_count": int(obj_ids.size),
                "propagation_runtime_sec": float(prop_sec),
                "propagation_video_scope": str(prop_scope),
                "propagation_frame_offset": int(prop_frame_offset),
                "propagation_state_num_frames_override": (
                    int(prop_state_num_frames) if prop_state_num_frames is not None else None
                ),
                "propagation_clear_cached_features_after_init": bool(prop_clear_cache),
                "propagation_reuse_video_state_template": bool(video_state_template is not None and str(prop_scope) == "full"),
                "future_output_mask_count": int(future_output_count),
                "propagation_chunk_runtime_records": propagation_chunk_runtime_records,
            }
            prop_record.update(frame0_hybrid_record)
            propagation_records.append(prop_record)

            if acc_masks[chunk_idx]:
                current_ids_pre = np.asarray(acc_ids[chunk_idx], dtype=np.int64)
                current_masks_pre = np.stack(acc_masks[chunk_idx], axis=0).astype(bool)
                current_masks_all, keep = disjoin_keep_order(
                    current_masks_pre,
                    h,
                    w,
                    empty_ratio=float(args.empty_ratio),
                )
                current_masks = current_masks_all[keep]
                current_ids = current_ids_pre[keep]
            else:
                current_ids = np.zeros((0,), dtype=np.int64)
                current_masks = np.zeros((0, h, w), dtype=bool)

        current_union = np.any(current_masks, axis=0) if current_masks.size else np.zeros((h, w), dtype=bool)
        uncovered = uncovered_from_masks(current_masks, h, w)
        core, envelope = load_alltracker_masks(alltracker_dir, frame_id, h, w)
        candidate = build_candidate_mask(
            variant=str(cli.variant),
            uncovered=uncovered,
            core=core,
            envelope=envelope,
            dilation_px=int(cli.provider_dilation_px),
        )
        if str(cli.variant) == "p4_temporal_persistent_definite":
            persistent_candidate_hits[candidate] = np.minimum(persistent_candidate_hits[candidate] + 1, 255)
            candidate = candidate & (persistent_candidate_hits >= int(cli.temporal_persistence_min_frames))
        else:
            persistent_candidate_hits[candidate] = np.minimum(persistent_candidate_hits[candidate] + 1, 255)

        candidate_before_anchor = candidate.astype(bool)
        candidate_area_before_anchor = int(np.count_nonzero(candidate_before_anchor))
        birth_anchor_period = max(int(cli.birth_anchor_period), 1)
        birth_anchor_offset = int(cli.birth_anchor_offset) % birth_anchor_period
        birth_anchor_scheduled_allowed = bool(
            birth_anchor_period <= 1 or (int(chunk_idx) - birth_anchor_offset) % birth_anchor_period == 0
        )
        birth_anchor_force_candidate_area_thresh = max(int(cli.birth_anchor_force_candidate_area_thresh), 0)
        birth_anchor_force_max_events = max(int(cli.birth_anchor_force_max_events), 0)
        birth_anchor_force_event_budget_available = bool(
            birth_anchor_force_max_events <= 0 or birth_anchor_forced_event_count < birth_anchor_force_max_events
        )
        birth_anchor_forced_by_candidate_area = bool(
            (not birth_anchor_scheduled_allowed)
            and birth_anchor_force_candidate_area_thresh > 0
            and candidate_area_before_anchor >= birth_anchor_force_candidate_area_thresh
            and birth_anchor_force_event_budget_available
        )
        if birth_anchor_forced_by_candidate_area:
            birth_anchor_forced_event_count += 1
        birth_anchor_allowed = bool(birth_anchor_scheduled_allowed or birth_anchor_forced_by_candidate_area)
        if not birth_anchor_allowed:
            candidate_for_birth = np.zeros_like(candidate_before_anchor, dtype=bool)
        else:
            candidate_for_birth = candidate_before_anchor

        candidate_stats = component_area_stats(candidate_for_birth, int(cli.min_component_area))
        point_seed = stable_seed(args.seed, args.scene_id, frame_id, str(cli.variant), "phase6-spec-gap")
        points_yx, point_meta = sample_component_adaptive_points_yx(
            candidate_for_birth,
            max_points=int(cli.max_points_per_frame),
            min_component_area=int(cli.min_component_area),
            base_points_per_component=int(cli.base_points_per_component),
            area_per_extra_point=int(cli.area_per_extra_point),
            max_points_per_component=int(cli.max_points_per_component),
            seed=int(point_seed),
        )

        birth_region_mask_source = "none_plan_no_hard_clip"
        birth_region_mask = None
        if bool(cli.enable_birth_region_mask) and not bool(cli.disable_birth_region_mask):
            birth_region_mask_source = "candidate_for_birth_hard_clip"
            birth_region_mask = candidate_for_birth

        birth_t0 = time.time()
        component_fallback_mode = str(cli.component_fallback_mode)
        if component_fallback_mode == "skip_sam":
            raw_birth_masks, component_records = component_masks_from_candidate(
                candidate_for_birth,
                min_area=int(cli.component_fallback_min_area),
                max_components=int(cli.component_fallback_max_components),
            )
            birth_stats = {
                "choice_policy": str(cli.choice_policy),
                "raw_multimask_option_count": 0,
                "prompt_with_good_mask_count": 0,
                "candidate_supported_option_count": 0,
                "prompt_with_candidate_supported_mask_count": 0,
                "pre_nms_mask_count": 0,
                "post_disjoint_mask_count": int(raw_birth_masks.shape[0]),
                "apply_box_nms": False,
                "nms_score_type": "component_fallback",
                "component_fallback_prompt_decoder_skipped": True,
                "component_fallback_raw_count": int(raw_birth_masks.shape[0]),
                "component_fallback_component_records": component_records[:64],
            }
        elif int(points_yx.shape[0]) > 0:
            if str(cli.choice_policy) in {
                "smallest_valid_candidate_supported",
                "max_candidate_support_valid_mask_per_point",
            }:
                raw_birth_masks, birth_stats = run_sam2_point_segment_choice_candidate_support(
                    segmentor,
                    rgbs[chunk_idx],
                    points_yx=points_yx,
                    support_mask=candidate_for_birth,
                    points_per_batch=int(args.points_per_batch),
                    choice_policy=str(cli.choice_policy),
                    iou_threshold=float(cli.pred_iou_thresh),
                    stability_threshold=float(cli.stability_score_thresh),
                    stability_score_offset=float(args.stability_score_offset),
                    model_mask_thresh=float(args.model_mask_thresh),
                    box_nms_thresh=float(args.box_nms_thresh),
                    empty_ratio=float(args.empty_ratio),
                    apply_box_nms=bool(cli.apply_box_nms),
                    nms_score_type=str(cli.nms_score_type),
                    support_min_area=int(cli.min_candidate_touch_area),
                    support_min_ratio=float(cli.min_candidate_touch_ratio),
                )
            else:
                raw_birth_masks, birth_stats = run_sam2_point_segment_choice(
                    segmentor,
                    rgbs[chunk_idx],
                    points_yx=points_yx,
                    region_mask=birth_region_mask,
                    points_per_batch=int(args.points_per_batch),
                    choice_policy=str(cli.choice_policy),
                    iou_threshold=float(cli.pred_iou_thresh),
                    stability_threshold=float(cli.stability_score_thresh),
                    stability_score_offset=float(args.stability_score_offset),
                    model_mask_thresh=float(args.model_mask_thresh),
                    box_nms_thresh=float(args.box_nms_thresh),
                    empty_ratio=float(args.empty_ratio),
                    apply_box_nms=bool(cli.apply_box_nms),
                    nms_score_type=str(cli.nms_score_type),
                )
        else:
            raw_birth_masks = np.zeros((0, h, w), dtype=bool)
            birth_stats = {
                "choice_policy": str(cli.choice_policy),
                "raw_multimask_option_count": 0,
                "prompt_with_good_mask_count": 0,
                "pre_nms_mask_count": 0,
                "post_disjoint_mask_count": 0,
                "apply_box_nms": bool(cli.apply_box_nms),
                "nms_score_type": str(cli.nms_score_type),
            }
        birth_stats["region_mask_source"] = birth_region_mask_source
        birth_stats["region_mask_area"] = (
            int(np.count_nonzero(birth_region_mask)) if birth_region_mask is not None else None
        )
        birth_decode_sec = time.time() - birth_t0
        total_birth_decode_sec += birth_decode_sec

        filtered_birth_masks, filter_records = filter_birth_masks(
            raw_birth_masks,
            candidate=candidate_for_birth,
            current_union=current_union,
            core=core,
            min_birth_mask_area=int(cli.min_birth_mask_area),
            min_candidate_touch_area=int(cli.min_candidate_touch_area),
            min_candidate_touch_ratio=float(cli.min_candidate_touch_ratio),
            max_existing_overlap_ratio=float(cli.max_existing_overlap_ratio),
            max_core_overlap_ratio=float(cli.max_core_overlap_ratio),
        )
        prompt_repair_mode = str(cli.prompt_repair_mode)
        prompt_repair_record: dict[str, Any] = {
            "mode": prompt_repair_mode,
            "applied": False,
            "reason": "disabled" if prompt_repair_mode == "disabled" else "condition_not_met",
            "base_raw_birth_mask_count": int(raw_birth_masks.shape[0]),
            "base_filtered_birth_mask_count": int(filtered_birth_masks.shape[0]),
            "repair_raw_birth_mask_count": 0,
            "repair_filtered_birth_mask_count": 0,
        }
        repair_condition = False
        if prompt_repair_mode == "always":
            repair_condition = True
        elif prompt_repair_mode == "on_raw_empty":
            repair_condition = int(raw_birth_masks.shape[0]) == 0
        elif prompt_repair_mode == "on_filtered_empty":
            repair_condition = int(filtered_birth_masks.shape[0]) == 0
        elif prompt_repair_mode == "on_raw_or_filtered_empty":
            repair_condition = int(raw_birth_masks.shape[0]) == 0 or int(filtered_birth_masks.shape[0]) == 0
        elif prompt_repair_mode != "disabled":
            raise ValueError(f"unsupported prompt_repair_mode={prompt_repair_mode}")
        if (
            prompt_repair_mode != "disabled"
            and repair_condition
            and int(np.count_nonzero(candidate_for_birth)) > 0
        ):
            repair_seed = stable_seed(args.seed, args.scene_id, frame_id, str(cli.variant), "phase6-box-point-repair")
            repair_raw_birth_masks, repair_stats = run_sam2_box_point_component_repair(
                segmentor,
                rgbs[chunk_idx],
                candidate=candidate_for_birth,
                negative_mask=(current_union | core),
                min_component_area=int(cli.prompt_repair_min_component_area),
                max_components=int(cli.prompt_repair_max_components),
                positive_points_per_component=int(cli.prompt_repair_positive_points_per_component),
                negative_points_per_component=int(cli.prompt_repair_negative_points_per_component),
                box_expand_px=int(cli.prompt_repair_box_expand_px),
                seed=int(repair_seed),
                iou_threshold=float(cli.prompt_repair_pred_iou_thresh),
                stability_threshold=float(cli.prompt_repair_stability_score_thresh),
                stability_score_offset=float(args.stability_score_offset),
                model_mask_thresh=float(args.model_mask_thresh),
                box_nms_thresh=float(args.box_nms_thresh),
                empty_ratio=float(args.empty_ratio),
                apply_box_nms=bool(cli.prompt_repair_apply_box_nms),
                nms_score_type=str(cli.nms_score_type),
                support_min_area=int(cli.min_candidate_touch_area),
                support_min_ratio=float(cli.min_candidate_touch_ratio),
                min_component_completion_ratio=float(cli.prompt_repair_min_component_completion_ratio),
            )
            repair_filtered_birth_masks, repair_filter_records = filter_birth_masks(
                repair_raw_birth_masks,
                candidate=candidate_for_birth,
                current_union=current_union,
                core=core,
                min_birth_mask_area=int(cli.min_birth_mask_area),
                min_candidate_touch_area=int(cli.min_candidate_touch_area),
                min_candidate_touch_ratio=float(cli.min_candidate_touch_ratio),
                max_existing_overlap_ratio=float(cli.max_existing_overlap_ratio),
                max_core_overlap_ratio=float(cli.max_core_overlap_ratio),
            )
            if repair_raw_birth_masks.size:
                raw_birth_masks = (
                    np.concatenate([raw_birth_masks.astype(bool), repair_raw_birth_masks.astype(bool)], axis=0)
                    if raw_birth_masks.size
                    else repair_raw_birth_masks.astype(bool)
                )
            if repair_filtered_birth_masks.size:
                filtered_birth_masks = (
                    np.concatenate([filtered_birth_masks.astype(bool), repair_filtered_birth_masks.astype(bool)], axis=0)
                    if filtered_birth_masks.size
                    else repair_filtered_birth_masks.astype(bool)
                )
            filter_records.extend(
                [
                    {
                        **record,
                        "source": "component_box_point_prompt_repair",
                    }
                    for record in repair_filter_records
                ]
            )
            prompt_repair_record = {
                "mode": prompt_repair_mode,
                "applied": True,
                "reason": "triggered_by_failure_condition",
                "base_raw_birth_mask_count": int(prompt_repair_record["base_raw_birth_mask_count"]),
                "base_filtered_birth_mask_count": int(prompt_repair_record["base_filtered_birth_mask_count"]),
                "repair_raw_birth_mask_count": int(repair_raw_birth_masks.shape[0]),
                "repair_filtered_birth_mask_count": int(repair_filtered_birth_masks.shape[0]),
                "combined_raw_birth_mask_count": int(raw_birth_masks.shape[0]),
                "combined_filtered_birth_mask_count": int(filtered_birth_masks.shape[0]),
                "repair_stats": repair_stats,
                "repair_filter_records": repair_filter_records[:64],
            }
        elif prompt_repair_mode != "disabled" and repair_condition:
            prompt_repair_record["reason"] = "condition_met_but_candidate_empty"
        if component_fallback_mode in {"when_empty", "always"}:
            filtered_birth_masks, component_fallback_record = append_component_fallback_births(
                filtered_birth_masks=filtered_birth_masks,
                candidate=candidate_for_birth,
                current_union=current_union,
                core=core,
                mode=("when_empty" if component_fallback_mode == "when_empty" else "always"),
                min_component_area=int(cli.component_fallback_min_area),
                max_components=int(cli.component_fallback_max_components),
                min_birth_mask_area=int(cli.min_birth_mask_area),
                min_candidate_touch_area=int(cli.min_candidate_touch_area),
                min_candidate_touch_ratio=float(cli.min_candidate_touch_ratio),
                max_existing_overlap_ratio=float(cli.max_existing_overlap_ratio),
                max_core_overlap_ratio=float(cli.max_core_overlap_ratio),
            )
        else:
            component_fallback_record = {
                "mode": component_fallback_mode,
                "applied": bool(component_fallback_mode == "skip_sam"),
                "base_filtered_birth_mask_count": int(filtered_birth_masks.shape[0]) if filtered_birth_masks.size else 0,
                "candidate_component_raw_count": int(raw_birth_masks.shape[0]) if component_fallback_mode == "skip_sam" else 0,
                "candidate_component_filtered_count": int(filtered_birth_masks.shape[0]) if component_fallback_mode == "skip_sam" else 0,
                "output_birth_mask_count": int(filtered_birth_masks.shape[0]) if filtered_birth_masks.size else 0,
                "audit_note": "component fallback disabled or prompt decoder skipped before standard filtering",
            }
        if filtered_birth_masks.size and int(cli.max_births_per_frame) > 0:
            areas = np.count_nonzero(filtered_birth_masks.reshape(filtered_birth_masks.shape[0], -1), axis=1)
            keep_idx = np.argsort(areas)[::-1][: int(cli.max_births_per_frame)]
            keep_idx.sort()
            accepted_birth_masks = filtered_birth_masks[keep_idx]
        else:
            accepted_birth_masks = filtered_birth_masks
        accepted_birth_ids = np.arange(next_obj_id, next_obj_id + int(accepted_birth_masks.shape[0]), dtype=np.int64)
        next_obj_id += int(accepted_birth_ids.size)
        to_prop_ids[chunk_idx] = accepted_birth_ids
        to_prop_masks[chunk_idx] = accepted_birth_masks
        stream_add_birth_sec = 0.0
        stream_state_repairs: list[dict[str, Any]] = []
        stream_state_prune_after_birth: dict[str, Any] = {"enabled": False, "reason": "no_stream_birth_add"}
        if stream_state is not None and accepted_birth_ids.size:
            stream_add_t0 = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                add_masks_to_stream_state(
                    tracker_model,
                    stream_state,
                    tracker=str(args.tracker_backend),
                    frame_idx=int(chunk_idx),
                    obj_ids=accepted_birth_ids,
                    masks=accepted_birth_masks,
                )
                stream_state_repairs = reconcile_stream_state_object_count(
                    tracker_model,
                    stream_state,
                    repair_mode=str(cli.stream_state_repair_mode),
                )
            stream_add_birth_sec = time.time() - stream_add_t0
            total_tracking_sec += stream_add_birth_sec
            stream_state_prune_after_birth = prune_stream_noncond_memory(
                stream_state,
                current_frame_idx=int(chunk_idx),
                keep_window=int(cli.stream_prune_noncond_keep_window),
            )

        for local_idx, (obj_id, mask) in enumerate(zip(accepted_birth_ids.tolist(), accepted_birth_masks.astype(bool), strict=False)):
            mask_path = birth_mask_dir / f"frame_{frame_id:06d}_obj_{int(obj_id):06d}_{cli.variant}.png"
            birth_mask_write_t0 = time.time()
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            total_birth_mask_write_sec += float(time.time() - birth_mask_write_t0)
            birth_records.append(
                {
                    "scene_id": str(args.scene_id),
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "obj_id": int(obj_id),
                    "local_birth_index": int(local_idx),
                    "source": str(cli.variant),
                    "mask_path": str(mask_path),
                    "mask_area": int(np.count_nonzero(mask)),
                }
            )

        frame_out_ids, frame_out_masks = write_frame_output(
            chunk_idx,
            current_ids,
            current_masks,
            accepted_birth_ids,
            accepted_birth_masks,
            {
                "source": str(cli.variant),
                "current_propagated_object_count": int(current_ids.size),
                "candidate_area": int(np.count_nonzero(candidate_for_birth)),
                "candidate_area_before_anchor_selection": int(candidate_area_before_anchor),
                "birth_anchor_period": int(birth_anchor_period),
                "birth_anchor_offset": int(birth_anchor_offset),
                "birth_anchor_scheduled_allowed": bool(birth_anchor_scheduled_allowed),
                "birth_anchor_allowed": bool(birth_anchor_allowed),
                "birth_anchor_forced_by_candidate_area": bool(birth_anchor_forced_by_candidate_area),
                "birth_anchor_force_candidate_area_thresh": int(birth_anchor_force_candidate_area_thresh),
                "birth_anchor_force_max_events": int(birth_anchor_force_max_events),
                "birth_anchor_forced_event_count": int(birth_anchor_forced_event_count),
                "birth_region_mask_source": birth_region_mask_source,
                "birth_region_mask_area": int(np.count_nonzero(birth_region_mask)) if birth_region_mask is not None else None,
                "candidate_component_count": int(candidate_stats["component_count"]),
                "candidate_kept_component_count": int(candidate_stats["kept_component_count"]),
                "candidate_max_component_area": int(candidate_stats["max_component_area"]),
                "candidate_area_ge_min_total": int(candidate_stats["area_ge_min_total"]),
                "candidate_point_count": int(points_yx.shape[0]),
                "candidate_sampling_meta": point_meta,
                "birth_decode_runtime_sec": float(birth_decode_sec),
                "stream_add_birth_runtime_sec": float(stream_add_birth_sec),
                "stream_state_repair_count": int(len(stream_state_repairs)),
                "stream_state_repairs": stream_state_repairs[:16],
                "stream_state_prune_after_birth": stream_state_prune_after_birth,
                "raw_birth_mask_count": int(raw_birth_masks.shape[0]),
                "accepted_birth_mask_count": int(accepted_birth_ids.size),
                "birth_stats": birth_stats,
                "prompt_repair": prompt_repair_record,
                "birth_filter_records": filter_records[:64],
                "component_fallback": component_fallback_record,
            },
        )
        if scheduler_mode == "rolling_one_step":
            rolling_prev_ids = frame_out_ids
            rolling_prev_masks = frame_out_masks
        print(
            json.dumps(
                {
                    "scene_id": str(args.scene_id),
                    "variant": str(cli.variant),
                    "chunk_frame_index": int(chunk_idx),
                    "frame_id": int(frame_id),
                    "candidate_area": int(np.count_nonzero(candidate_for_birth)),
                    "candidate_area_before_anchor_selection": int(candidate_area_before_anchor),
                    "birth_anchor_period": int(birth_anchor_period),
                    "birth_anchor_offset": int(birth_anchor_offset),
                    "birth_anchor_scheduled_allowed": bool(birth_anchor_scheduled_allowed),
                    "birth_anchor_allowed": bool(birth_anchor_allowed),
                    "birth_anchor_forced_by_candidate_area": bool(birth_anchor_forced_by_candidate_area),
                    "birth_anchor_force_candidate_area_thresh": int(birth_anchor_force_candidate_area_thresh),
                    "birth_anchor_force_max_events": int(birth_anchor_force_max_events),
                    "birth_anchor_forced_event_count": int(birth_anchor_forced_event_count),
                    "points": int(points_yx.shape[0]),
                    "raw_births": int(raw_birth_masks.shape[0]),
                    "accepted_births": int(accepted_birth_ids.size),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    if stream_state is not None:
        try:
            tracker_model.reset_state(stream_state)
        except Exception:
            pass
        try:
            stream_state.clear()
        except Exception:
            pass

    sheet_paths: list[str] = []
    sheet_write_sec = 0.0
    if not bool(cli.skip_chunk_sheets):
        if bool(cli.skip_chunk_overlays):
            raise ValueError("--skip-chunk-sheets is required when --skip-chunk-overlays is used")
        sheet_write_t0 = time.time()
        for start in range(0, len(overlay_paths), 8):
            part = overlay_paths[start : start + 8]
            end = start + len(part) - 1
            sheet_path = sheet_dir / f"phase6_{cli.variant}_{args.scene_id}_frames_{start:02d}_{end:02d}_4x2.jpg"
            make_sheet_grid(part, sheet_path, int(args.sheet_cell_width), cols=4)
            sheet_paths.append(str(sheet_path))
        sheet_write_sec = float(time.time() - sheet_write_t0)
    video_path = video_dir_out / f"phase6_{cli.variant}_{args.scene_id}_chunk0.mp4"
    video_write_sec = 0.0
    if not bool(cli.skip_chunk_video):
        if bool(cli.skip_chunk_overlays):
            raise ValueError("--skip-chunk-video is required when --skip-chunk-overlays is used")
        video_write_t0 = time.time()
        write_video(overlay_paths, video_path, fps=float(args.fps))
        video_write_sec = float(time.time() - video_write_t0)

    total_sec = time.time() - total_t0
    peak_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
    x1_runtime = x1_summary.get("total_runtime_sec")
    x0_runtime = x0_summary.get("total_runtime_sec")
    missing_x1_frame_ids = [int(row["frame_id"]) for row in records if not bool(row.get("x1_diagnostic_available"))]
    x1_diagnostics_complete = not missing_x1_frame_ids
    x1_ious = [
        float(row["x1_foreground_metrics"]["fg_iou"])
        for row in records
        if isinstance(row.get("x1_foreground_metrics"), dict)
    ]
    x1_precisions = [
        float(row["x1_foreground_metrics"]["fg_precision"])
        for row in records
        if isinstance(row.get("x1_foreground_metrics"), dict)
    ]
    x1_recalls = [
        float(row["x1_foreground_metrics"]["fg_recall"])
        for row in records
        if isinstance(row.get("x1_foreground_metrics"), dict)
    ]
    missing_x0_frame_ids = [int(row["frame_id"]) for row in records if not bool(row.get("x0_diagnostic_available"))]
    x0_diagnostics_complete = not missing_x0_frame_ids
    x0_ious = [
        float(row["x0_foreground_metrics"]["fg_iou"])
        for row in records
        if isinstance(row.get("x0_foreground_metrics"), dict)
    ]
    birth_count_total = int(sum(int(row.get("birth_count", 0)) for row in records))
    summary = {
        "schema_version": "stream4d_v105_phase6_speculative_gap_birth_summary_v1",
        "scene_id": str(args.scene_id),
        "variant": str(cli.variant),
        "config_path": str(config_path),
        "frame_ids": [int(v) for v in frame_ids],
        "frame_count": int(len(frame_ids)),
        "alltracker_dir": str(alltracker_dir),
        "alltracker_summary_sha256": sha256_file(alltracker_summary_path),
        "alltracker_diagnostic_only_input_boundary": "uses only core/envelope masks; foreground_outside_envelope is not read",
        "frame0_seed_meta": seed_meta,
        "x0_summary": str(resolve_path(cli.x0_summary)),
        "x0_summary_sha256": sha256_file(resolve_path(cli.x0_summary)),
        "x1_summary": str(resolve_path(cli.x1_summary)),
        "x1_summary_sha256": sha256_file(resolve_path(cli.x1_summary)),
        "reference_x1_total_runtime_sec": float(x1_runtime) if x1_runtime is not None else None,
        "reference_x0_total_runtime_sec": float(x0_runtime) if x0_runtime is not None else None,
        "x1_diagnostics_complete": bool(x1_diagnostics_complete),
        "x1_diagnostic_available_frame_count": int(len(x1_ious)),
        "x1_diagnostic_missing_frame_count": int(len(missing_x1_frame_ids)),
        "x1_diagnostic_missing_frame_ids": missing_x1_frame_ids[:200],
        "x1_diagnostic_missing_frame_ids_truncated": bool(len(missing_x1_frame_ids) > 200),
        "x1_runtime_ratio_valid": bool(x1_diagnostics_complete),
        "x0_diagnostics_complete": bool(x0_diagnostics_complete),
        "x0_diagnostic_available_frame_count": int(len(x0_ious)),
        "x0_diagnostic_missing_frame_count": int(len(missing_x0_frame_ids)),
        "x0_diagnostic_missing_frame_ids": missing_x0_frame_ids[:200],
        "x0_diagnostic_missing_frame_ids_truncated": bool(len(missing_x0_frame_ids) > 200),
        "x0_runtime_ratio_valid": bool(x0_diagnostics_complete),
        "setup_sec": float(setup_sec),
        "total_runtime_sec": float(total_sec),
        "total_tracking_runtime_sec": float(total_tracking_sec),
        "total_birth_decode_runtime_sec": float(total_birth_decode_sec),
        "output_timing_sec": {
            "frame_output_total_sec": float(total_frame_output_sec),
            "label_build_sec": float(total_label_build_sec),
            "label_write_sec": float(total_label_write_sec),
            "diagnostic_load_metric_sec": float(total_diagnostic_sec),
            "overlay_render_write_sec": float(total_overlay_write_sec),
            "birth_mask_write_sec": float(total_birth_mask_write_sec),
            "sheet_write_sec": float(sheet_write_sec),
            "video_write_sec": float(video_write_sec),
            "artifact_write_sec": float(
                total_label_write_sec
                + total_overlay_write_sec
                + total_birth_mask_write_sec
                + sheet_write_sec
                + video_write_sec
            ),
            "non_model_output_path_sec": float(total_frame_output_sec + total_birth_mask_write_sec + sheet_write_sec + video_write_sec),
        },
        "reuse_video_state_template": bool(cli.reuse_video_state_template),
        "video_state_template_init_sec": float(video_state_template_init_sec),
        "chunk_visual_outputs": {
            "skip_chunk_overlays": bool(cli.skip_chunk_overlays),
            "skip_chunk_sheets": bool(cli.skip_chunk_sheets),
            "skip_chunk_video": bool(cli.skip_chunk_video),
            "overlay_count": int(len(overlay_paths)),
            "sheet_count": int(len(sheet_paths)),
            "chunk_video_exists": bool(video_path.exists() and video_path.stat().st_size > 0),
        },
        "runtime_ratio_vs_x1": (
            float(total_sec) / float(x1_runtime)
            if x1_diagnostics_complete and x1_runtime is not None and float(x1_runtime) > 0.0
            else None
        ),
        "runtime_ratio_vs_x0": (
            float(total_sec) / float(x0_runtime)
            if x0_diagnostics_complete and x0_runtime is not None and float(x0_runtime) > 0.0
            else None
        ),
        "latency_gate_le_0p7_x1": (
            bool(float(total_sec) <= 0.7 * float(x1_runtime))
            if x1_diagnostics_complete and x1_runtime is not None and float(x1_runtime) > 0.0
            else None
        ),
        "candidate_main_gate_le_0p5_x1": (
            bool(float(total_sec) <= 0.5 * float(x1_runtime))
            if x1_diagnostics_complete and x1_runtime is not None and float(x1_runtime) > 0.0
            else None
        ),
        "single_gpu_vram_gate_le_20gb": bool(float(peak_mb) <= 20.0 * 1024.0),
        "peak_cuda_memory_mb": float(peak_mb),
        "offload_video_to_cpu": bool(args.offload_video_to_cpu),
        "offload_state_to_cpu": bool(args.offload_state_to_cpu),
        "propagation_chunk_size": int(args.propagation_chunk_size),
        "propagation_video_scope": str(cli.propagation_video_scope),
        "tracker_num_maskmem_override": (
            int(cli.tracker_num_maskmem_override) if int(cli.tracker_num_maskmem_override) > 0 else None
        ),
        "tracker_num_maskmem_original": (
            int(tracker_num_maskmem_original) if tracker_num_maskmem_original is not None else None
        ),
        "tracker_num_maskmem_effective": (
            int(getattr(tracker_model, "num_maskmem"))
            if getattr(tracker_model, "num_maskmem", None) is not None
            else None
        ),
        "tracker_max_obj_ptrs_in_encoder_override": (
            int(cli.tracker_max_obj_ptrs_in_encoder_override)
            if int(cli.tracker_max_obj_ptrs_in_encoder_override) > 0
            else None
        ),
        "tracker_max_obj_ptrs_in_encoder_original": (
            int(tracker_max_obj_ptrs_in_encoder_original)
            if tracker_max_obj_ptrs_in_encoder_original is not None
            else None
        ),
        "tracker_max_obj_ptrs_in_encoder_effective": (
            int(getattr(tracker_model, "max_obj_ptrs_in_encoder"))
            if getattr(tracker_model, "max_obj_ptrs_in_encoder", None) is not None
            else None
        ),
        "scheduler_mode": str(cli.scheduler_mode),
        "stream_state_repair_mode": str(cli.stream_state_repair_mode),
        "frame0_hybrid_small_num_maskmem": (
            int(cli.frame0_hybrid_small_num_maskmem)
            if int(cli.frame0_hybrid_small_num_maskmem) > 0
            else None
        ),
        "frame0_hybrid_full_topk": (
            int(cli.frame0_hybrid_full_topk) if int(cli.frame0_hybrid_full_topk) > 0 else None
        ),
        "frame0_hybrid_full_min_area": (
            int(cli.frame0_hybrid_full_min_area) if int(cli.frame0_hybrid_full_min_area) > 0 else None
        ),
        "stream_max_cond_frames_in_attn": (
            int(cli.stream_max_cond_frames_in_attn) if int(cli.stream_max_cond_frames_in_attn) >= 0 else None
        ),
        "stream_max_cond_frames_in_attn_original": (
            int(stream_max_cond_frames_in_attn_original)
            if stream_max_cond_frames_in_attn_original is not None
            else None
        ),
        "stream_prune_noncond_keep_window": (
            int(cli.stream_prune_noncond_keep_window) if int(cli.stream_prune_noncond_keep_window) >= 0 else None
        ),
        "stream_clear_noncond_mem_around_input": bool(cli.stream_clear_noncond_mem_around_input),
        "stream_clear_noncond_mem_around_input_original": (
            bool(stream_clear_noncond_mem_around_input_original)
            if stream_clear_noncond_mem_around_input_original is not None
            else None
        ),
        "stream_clear_noncond_mem_for_multi_obj_original": (
            bool(stream_clear_noncond_mem_for_multi_obj_original)
            if stream_clear_noncond_mem_for_multi_obj_original is not None
            else None
        ),
        "use_video_feature_bank": bool(cli.use_video_feature_bank),
        "video_feature_bank": serializable_feature_bank_summary(feature_bank_info),
        "bf16_autocast_paths": [
            "SAM2 point-prompt decode via run_sam2_point_segment_choice",
            "SAM2 propagation via propagate_new_masks_chunked",
            "video feature bank build via install_video_feature_bank_patch",
        ],
        "variant_parameters": {
            "min_component_area": int(cli.min_component_area),
            "max_points_per_frame": int(cli.max_points_per_frame),
            "base_points_per_component": int(cli.base_points_per_component),
            "area_per_extra_point": int(cli.area_per_extra_point),
            "max_points_per_component": int(cli.max_points_per_component),
            "min_birth_mask_area": int(cli.min_birth_mask_area),
            "min_candidate_touch_area": int(cli.min_candidate_touch_area),
            "min_candidate_touch_ratio": float(cli.min_candidate_touch_ratio),
            "max_existing_overlap_ratio": float(cli.max_existing_overlap_ratio),
            "max_core_overlap_ratio": float(cli.max_core_overlap_ratio),
            "max_births_per_frame": int(cli.max_births_per_frame),
            "provider_dilation_px": int(cli.provider_dilation_px),
            "temporal_persistence_min_frames": int(cli.temporal_persistence_min_frames),
            "birth_anchor_period": int(cli.birth_anchor_period),
            "birth_anchor_offset": int(cli.birth_anchor_offset),
            "birth_anchor_force_candidate_area_thresh": int(cli.birth_anchor_force_candidate_area_thresh),
            "birth_anchor_force_max_events": int(cli.birth_anchor_force_max_events),
            "enable_birth_region_mask": bool(cli.enable_birth_region_mask),
            "disable_birth_region_mask": bool(cli.disable_birth_region_mask),
            "choice_policy": str(cli.choice_policy),
            "pred_iou_thresh": float(cli.pred_iou_thresh),
            "stability_score_thresh": float(cli.stability_score_thresh),
            "apply_box_nms": bool(cli.apply_box_nms),
            "nms_score_type": str(cli.nms_score_type),
            "prompt_repair_mode": str(cli.prompt_repair_mode),
            "prompt_repair_min_component_area": int(cli.prompt_repair_min_component_area),
            "prompt_repair_max_components": int(cli.prompt_repair_max_components),
            "prompt_repair_positive_points_per_component": int(cli.prompt_repair_positive_points_per_component),
            "prompt_repair_negative_points_per_component": int(cli.prompt_repair_negative_points_per_component),
            "prompt_repair_box_expand_px": int(cli.prompt_repair_box_expand_px),
            "prompt_repair_min_component_completion_ratio": float(cli.prompt_repair_min_component_completion_ratio),
            "prompt_repair_pred_iou_thresh": float(cli.prompt_repair_pred_iou_thresh),
            "prompt_repair_stability_score_thresh": float(cli.prompt_repair_stability_score_thresh),
            "prompt_repair_apply_box_nms": bool(cli.prompt_repair_apply_box_nms),
            "component_fallback_mode": str(cli.component_fallback_mode),
            "component_fallback_min_area": int(cli.component_fallback_min_area),
            "component_fallback_max_components": int(cli.component_fallback_max_components),
            "enable_frame0_residual_repair": bool(cli.enable_frame0_residual_repair),
            "frame0_residual_candidate_mode": str(cli.frame0_residual_candidate_mode),
            "frame0_residual_core_erosion_px": int(cli.frame0_residual_core_erosion_px),
            "frame0_residual_choice_policy": str(cli.frame0_residual_choice_policy),
            "frame0_residual_pred_iou_thresh": float(cli.frame0_residual_pred_iou_thresh),
            "frame0_residual_stability_score_thresh": float(cli.frame0_residual_stability_score_thresh),
            "frame0_residual_max_points": int(cli.frame0_residual_max_points),
            "frame0_residual_min_component_area": int(cli.frame0_residual_min_component_area),
            "frame0_residual_max_births": int(cli.frame0_residual_max_births),
            "frame0_residual_component_fallback_mode": str(cli.frame0_residual_component_fallback_mode),
            "frame0_residual_component_fallback_min_area": int(cli.frame0_residual_component_fallback_min_area),
            "frame0_residual_component_fallback_max_components": int(cli.frame0_residual_component_fallback_max_components),
            "frame0_residual_write_policy": (
                "write_only_base_uncovered_pixels"
                if str(cli.frame0_residual_candidate_mode) == "full_union_uncovered"
                else "write_only_eroded_core_uncovered_pixels"
            ),
            "scheduler_mode": str(cli.scheduler_mode),
            "stream_state_repair_mode": str(cli.stream_state_repair_mode),
            "tracker_num_maskmem_override": (
                int(cli.tracker_num_maskmem_override) if int(cli.tracker_num_maskmem_override) > 0 else None
            ),
            "tracker_max_obj_ptrs_in_encoder_override": (
                int(cli.tracker_max_obj_ptrs_in_encoder_override)
                if int(cli.tracker_max_obj_ptrs_in_encoder_override) > 0
                else None
            ),
            "frame0_hybrid_small_num_maskmem": (
                int(cli.frame0_hybrid_small_num_maskmem)
                if int(cli.frame0_hybrid_small_num_maskmem) > 0
                else None
            ),
            "frame0_hybrid_full_topk": (
                int(cli.frame0_hybrid_full_topk) if int(cli.frame0_hybrid_full_topk) > 0 else None
            ),
            "frame0_hybrid_full_min_area": (
                int(cli.frame0_hybrid_full_min_area) if int(cli.frame0_hybrid_full_min_area) > 0 else None
            ),
            "stream_max_cond_frames_in_attn": (
                int(cli.stream_max_cond_frames_in_attn) if int(cli.stream_max_cond_frames_in_attn) >= 0 else None
            ),
            "stream_prune_noncond_keep_window": (
                int(cli.stream_prune_noncond_keep_window) if int(cli.stream_prune_noncond_keep_window) >= 0 else None
            ),
            "stream_clear_noncond_mem_around_input": bool(cli.stream_clear_noncond_mem_around_input),
            "allow_missing_x0_diagnostics": bool(cli.allow_missing_x0_diagnostics),
            "allow_missing_x1_diagnostics": bool(cli.allow_missing_x1_diagnostics),
            "skip_chunk_overlays": bool(cli.skip_chunk_overlays),
            "skip_chunk_sheets": bool(cli.skip_chunk_sheets),
            "skip_chunk_video": bool(cli.skip_chunk_video),
        },
        "birth_count_total": int(birth_count_total),
        "mean_x1_foreground_iou": float(np.mean(x1_ious)) if x1_ious and x1_diagnostics_complete else None,
        "min_x1_foreground_iou": float(min(x1_ious)) if x1_ious and x1_diagnostics_complete else None,
        "mean_x1_foreground_precision": float(np.mean(x1_precisions)) if x1_precisions and x1_diagnostics_complete else None,
        "mean_x1_foreground_recall": float(np.mean(x1_recalls)) if x1_recalls and x1_diagnostics_complete else None,
        "mean_x1_foreground_iou_available_frames": float(np.mean(x1_ious)) if x1_ious else None,
        "mean_x0_foreground_iou": float(np.mean(x0_ious)) if x0_ious and x0_diagnostics_complete else None,
        "mean_x0_foreground_iou_available_frames": float(np.mean(x0_ious)) if x0_ious else None,
        "visual_gate_status": "manual_review_required",
        "records": records,
        "birth_records": birth_records,
        "propagation_records": propagation_records,
        "sheet_paths": sheet_paths,
        "video_path": str(video_path),
        "alltracker_summary_notes": alltracker_summary.get("notes", []),
    }
    summary_path = output_root / "phase6_speculative_gap_birth_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "video": str(video_path) if video_path.exists() else None,
                "sheets": sheet_paths,
                "chunk_visual_outputs": summary["chunk_visual_outputs"],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame0-birth-records", required=True)
    parser.add_argument("--alltracker-dir", required=True)
    parser.add_argument("--x0-summary", required=True)
    parser.add_argument("--x1-summary", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default=None)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--frame-ids", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--variant",
        choices=[
            "p3_definite_no_reconcile",
            "p3_uncertain_band_expansion",
            "p4_temporal_persistent_definite",
            "p4_anchor_period_definite",
            "p5_protected_core_definite",
        ],
        default="p3_definite_no_reconcile",
    )
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=False)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    parser.add_argument("--propagation-chunk-size", type=int, default=0)
    parser.add_argument("--propagation-video-scope", choices=["full", "suffix", "suffix_absolute"], default="full")
    parser.add_argument("--reuse-video-state-template", action="store_true", default=False)
    parser.add_argument(
        "--scheduler-mode",
        choices=["independent_anchor", "streaming_state", "rolling_one_step"],
        default="independent_anchor",
    )
    parser.add_argument("--stream-state-repair-mode", choices=["reconsolidate", "pad"], default="reconsolidate")
    parser.add_argument("--tracker-num-maskmem-override", type=int, default=-1)
    parser.add_argument("--tracker-max-obj-ptrs-in-encoder-override", type=int, default=-1)
    parser.add_argument("--frame0-hybrid-small-num-maskmem", type=int, default=-1)
    parser.add_argument("--frame0-hybrid-full-topk", type=int, default=-1)
    parser.add_argument("--frame0-hybrid-full-min-area", type=int, default=-1)
    parser.add_argument("--stream-max-cond-frames-in-attn", type=int, default=-1)
    parser.add_argument("--stream-prune-noncond-keep-window", type=int, default=-1)
    parser.add_argument("--stream-clear-noncond-mem-around-input", action="store_true", default=False)
    parser.add_argument("--use-video-feature-bank", action="store_true")
    parser.add_argument("--video-feature-bank-storage-device", default="cuda")
    parser.add_argument("--video-gpu-hot-window", type=int, default=0)
    parser.add_argument(
        "--skip-chunk-overlays",
        action="store_true",
        default=False,
        help="Write labels/summary only and skip per-frame Phase6 overlay JPEGs. Full-scene assembly can still render overlays from labels.",
    )
    parser.add_argument(
        "--skip-chunk-sheets",
        action="store_true",
        default=False,
        help="Skip Phase6 4x2 chunk sheet generation. Required when --skip-chunk-overlays is used.",
    )
    parser.add_argument(
        "--skip-chunk-video",
        action="store_true",
        default=False,
        help="Skip Phase6 chunk MP4 generation. Required when --skip-chunk-overlays is used.",
    )
    parser.add_argument("--min-component-area", type=int, default=200)
    parser.add_argument("--max-points-per-frame", type=int, default=64)
    parser.add_argument("--base-points-per-component", type=int, default=1)
    parser.add_argument("--area-per-extra-point", type=int, default=40000)
    parser.add_argument("--max-points-per-component", type=int, default=4)
    parser.add_argument("--min-birth-mask-area", type=int, default=128)
    parser.add_argument("--min-candidate-touch-area", type=int, default=16)
    parser.add_argument("--min-candidate-touch-ratio", type=float, default=0.001)
    parser.add_argument("--max-existing-overlap-ratio", type=float, default=0.98)
    parser.add_argument("--max-core-overlap-ratio", type=float, default=1.0)
    parser.add_argument("--max-births-per-frame", type=int, default=8)
    parser.add_argument("--provider-dilation-px", type=int, default=0)
    parser.add_argument("--temporal-persistence-min-frames", type=int, default=2)
    parser.add_argument("--birth-anchor-period", type=int, default=1)
    parser.add_argument("--birth-anchor-offset", type=int, default=0)
    parser.add_argument("--birth-anchor-force-candidate-area-thresh", type=int, default=0)
    parser.add_argument("--birth-anchor-force-max-events", type=int, default=0)
    parser.add_argument("--enable-birth-region-mask", action="store_true")
    parser.add_argument("--disable-birth-region-mask", action="store_true")
    parser.add_argument("--choice-policy", default="smallest_valid_mask_per_point")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.8)
    parser.add_argument("--apply-box-nms", action="store_true", default=False)
    parser.add_argument("--nms-score-type", choices=["pred_iou", "stability"], default="stability")
    parser.add_argument(
        "--prompt-repair-mode",
        choices=["disabled", "on_raw_empty", "on_filtered_empty", "on_raw_or_filtered_empty", "always"],
        default="disabled",
        help="Default-off SAM2 repair for candidate frames where the point prompt branch returns no raw or no filtered births. Uses component box plus positive/negative point prompts, then the normal birth filter.",
    )
    parser.add_argument("--prompt-repair-min-component-area", type=int, default=300)
    parser.add_argument("--prompt-repair-max-components", type=int, default=8)
    parser.add_argument("--prompt-repair-positive-points-per-component", type=int, default=3)
    parser.add_argument("--prompt-repair-negative-points-per-component", type=int, default=2)
    parser.add_argument("--prompt-repair-box-expand-px", type=int, default=4)
    parser.add_argument("--prompt-repair-min-component-completion-ratio", type=float, default=0.20)
    parser.add_argument("--prompt-repair-pred-iou-thresh", type=float, default=0.5)
    parser.add_argument("--prompt-repair-stability-score-thresh", type=float, default=0.5)
    parser.add_argument("--prompt-repair-apply-box-nms", action="store_true", default=False)
    parser.add_argument(
        "--component-fallback-mode",
        choices=["disabled", "when_empty", "always", "skip_sam"],
        default="disabled",
        help="Default-off repair/diagnostic: use connected components of the candidate gap as birth masks when point-prompt decoding is empty, always append them, or skip SAM prompt decoding entirely.",
    )
    parser.add_argument("--component-fallback-min-area", type=int, default=800)
    parser.add_argument("--component-fallback-max-components", type=int, default=4)
    parser.add_argument("--enable-frame0-residual-repair", action="store_true", default=False)
    parser.add_argument(
        "--frame0-residual-candidate-mode",
        choices=["full_union_uncovered", "eroded_core_uncovered"],
        default="full_union_uncovered",
        help="Default preserves old behavior. eroded_core_uncovered protects only an eroded frame0 seed core so SAM2 can rebirth flexible boundary/underseg regions.",
    )
    parser.add_argument("--frame0-residual-core-erosion-px", type=int, default=5)
    parser.add_argument("--frame0-residual-choice-policy", default="max_candidate_support_valid_mask_per_point")
    parser.add_argument("--frame0-residual-pred-iou-thresh", type=float, default=0.5)
    parser.add_argument("--frame0-residual-stability-score-thresh", type=float, default=0.5)
    parser.add_argument("--frame0-residual-max-points", type=int, default=96)
    parser.add_argument("--frame0-residual-min-component-area", type=int, default=800)
    parser.add_argument("--frame0-residual-base-points-per-component", type=int, default=1)
    parser.add_argument("--frame0-residual-area-per-extra-point", type=int, default=40000)
    parser.add_argument("--frame0-residual-max-points-per-component", type=int, default=12)
    parser.add_argument("--frame0-residual-min-birth-mask-area", type=int, default=100)
    parser.add_argument("--frame0-residual-min-candidate-touch-area", type=int, default=32)
    parser.add_argument("--frame0-residual-min-candidate-touch-ratio", type=float, default=0.01)
    parser.add_argument("--frame0-residual-max-existing-overlap-ratio", type=float, default=0.95)
    parser.add_argument("--frame0-residual-max-core-overlap-ratio", type=float, default=1.0)
    parser.add_argument("--frame0-residual-max-births", type=int, default=8)
    parser.add_argument(
        "--frame0-residual-component-fallback-mode",
        choices=["disabled", "when_empty", "always", "skip_sam"],
        default="disabled",
        help="Default-off frame0 repair/diagnostic for initial uncovered residuals; modes match --component-fallback-mode.",
    )
    parser.add_argument("--frame0-residual-component-fallback-min-area", type=int, default=1200)
    parser.add_argument("--frame0-residual-component-fallback-max-components", type=int, default=8)
    parser.add_argument(
        "--allow-missing-x0-diagnostics",
        action="store_true",
        default=False,
        help="Allow visualization/full-scene exploratory runs when X0 diagnostic labels do not cover every requested frame. Missing X0 metrics are written as unavailable; no X0 runtime ratio is reported.",
    )
    parser.add_argument(
        "--allow-missing-x1-diagnostics",
        action="store_true",
        default=False,
        help="Allow visualization/full-scene exploratory runs when X1 diagnostic labels do not cover every requested frame. Missing X1 metrics are written as unavailable; no X1 runtime ratio or latency gate is reported.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
