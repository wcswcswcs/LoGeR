#!/usr/bin/env python3
"""Merge obvious duplicate thing tracks in an existing sparse_masklets.pt.

This is an offline audit/postprocess tool. It does not run detection, SAM, or
semantic segmentation. It merges same-label thing tracks that overlap strongly
or reconnect after a small temporal gap with near-identical boxes, then writes a
new sparse file, review video, metrics JSON, and before/after contact sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    _unpack_mask_np,
    clone_sparse,
    coverage_stats,
    load_sparse,
    make_contact_sheet,
    parse_contact_frames,
    track_stats,
)
from run_video_masklet_front_end import _pack_mask_np, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge duplicate same-label thing tracks in sparse masklets.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--frame_cache_refresh", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--render_video", type=int, default=1)
    parser.add_argument("--render_contact_sheet", type=int, default=1)
    parser.add_argument("--fast_metrics", type=int, default=0)
    parser.add_argument("--canonicalize_vehicle_labels", type=int, default=0)
    parser.add_argument("--canonical_vehicle_labels", default="car,van,bus,truck,vehicle,trailer")
    parser.add_argument("--canonical_vehicle_output_label", default="car")
    parser.add_argument("--max_gap", type=int, default=8)
    parser.add_argument("--overlap_iou", type=float, default=0.50)
    parser.add_argument("--gap_box_iou", type=float, default=0.70)
    parser.add_argument("--center_dist", type=float, default=0.08)
    parser.add_argument("--min_track_frames", type=int, default=1)
    parser.add_argument("--reid_merge_tracks", type=int, default=0)
    parser.add_argument("--reid_max_gap", type=int, default=15)
    parser.add_argument("--reid_min_frames", type=int, default=4)
    parser.add_argument("--reid_min_hist_similarity", type=float, default=0.88)
    parser.add_argument("--reid_min_box_iou", type=float, default=0.50)
    parser.add_argument("--reid_max_center_dist", type=float, default=0.25)
    parser.add_argument("--reid_min_area_ratio", type=float, default=0.35)
    parser.add_argument("--reid_min_mask_pixels", type=int, default=40)
    parser.add_argument("--temporal_link_tracks", type=int, default=0)
    parser.add_argument("--temporal_link_labels", default="car,van,bus,truck,vehicle,trailer,motorcycle")
    parser.add_argument("--temporal_link_vehicle_compatible", type=int, default=0)
    parser.add_argument("--temporal_link_vehicle_labels", default="car,van,bus,truck,vehicle,trailer")
    parser.add_argument("--temporal_link_passes", type=int, default=1)
    parser.add_argument("--temporal_link_min_score", type=float, default=0.0)
    parser.add_argument("--temporal_link_cross_label_max_gap", type=int, default=10)
    parser.add_argument("--temporal_link_cross_label_min_box_iou", type=float, default=0.30)
    parser.add_argument("--temporal_link_cross_label_max_center_dist", type=float, default=0.70)
    parser.add_argument("--temporal_link_max_gap", type=int, default=25)
    parser.add_argument("--temporal_link_min_box_iou", type=float, default=0.18)
    parser.add_argument("--temporal_link_min_area_ratio", type=float, default=0.35)
    parser.add_argument("--temporal_link_max_center_dist", type=float, default=1.35)
    parser.add_argument("--temporal_link_max_pred_center_dist", type=float, default=1.00)
    parser.add_argument("--temporal_link_center_scale_floor", type=float, default=18.0)
    parser.add_argument("--temporal_link_long_gap_min_box_iou", type=float, default=0.30)
    parser.add_argument("--temporal_link_long_gap_max_center_dist", type=float, default=0.75)
    parser.add_argument("--temporal_link_long_gap", type=int, default=8)
    parser.add_argument("--overlap_support_merge_tracks", type=int, default=0)
    parser.add_argument("--overlap_support_vehicle_compatible", type=int, default=0)
    parser.add_argument("--overlap_support_vehicle_labels", default="car,van,bus,truck,vehicle,trailer")
    parser.add_argument("--overlap_support_min_common_frames", type=int, default=5)
    parser.add_argument("--overlap_support_min_duplicate_frames", type=int, default=3)
    parser.add_argument("--overlap_support_min_mask_iou", type=float, default=0.35)
    parser.add_argument("--overlap_support_min_mean_mask_iou", type=float, default=0.30)
    parser.add_argument("--overlap_support_min_p75_mask_iou", type=float, default=0.40)
    parser.add_argument("--overlap_support_min_mean_containment", type=float, default=0.50)
    parser.add_argument("--overlap_support_min_duplicate_ratio", type=float, default=0.0)
    parser.add_argument("--overlap_support_conflict_mask_iou", type=float, default=0.05)
    parser.add_argument("--overlap_support_conflict_containment", type=float, default=0.10)
    parser.add_argument("--overlap_support_max_conflict_frames", type=int, default=-1)
    parser.add_argument("--overlap_support_max_conflict_ratio", type=float, default=1.0)
    parser.add_argument("--trim_duplicate_frames", type=int, default=0)
    parser.add_argument("--trim_duplicate_max_track_frames", type=int, default=5)
    parser.add_argument("--trim_duplicate_mask_iou", type=float, default=0.35)
    parser.add_argument("--trim_duplicate_box_iou", type=float, default=0.75)
    parser.add_argument("--trim_duplicate_min_peer_frames", type=int, default=8)
    parser.add_argument("--trim_duplicate_min_peer_ratio", type=float, default=3.0)
    parser.add_argument("--trim_contained_duplicate_frames", type=int, default=0)
    parser.add_argument("--trim_contained_min_containment", type=float, default=0.90)
    parser.add_argument("--trim_contained_max_area_ratio", type=float, default=0.04)
    parser.add_argument("--trim_contained_min_peer_frames", type=int, default=8)
    parser.add_argument("--trim_endpoint_duplicate_frames", type=int, default=0)
    parser.add_argument("--trim_endpoint_margin", type=int, default=5)
    parser.add_argument("--trim_endpoint_min_mask_iou", type=float, default=0.35)
    parser.add_argument("--trim_endpoint_min_containment", type=float, default=0.65)
    parser.add_argument("--trim_endpoint_max_area_ratio", type=float, default=0.01)
    parser.add_argument("--trim_endpoint_min_peer_frames", type=int, default=8)
    parser.add_argument("--drop_short_duplicate_tracks", type=int, default=0)
    parser.add_argument("--drop_short_max_frames", type=int, default=3)
    parser.add_argument("--drop_duplicate_mask_iou", type=float, default=0.35)
    parser.add_argument("--drop_duplicate_box_iou", type=float, default=0.55)
    parser.add_argument("--drop_duplicate_containment", type=float, default=0.90)
    parser.add_argument("--drop_duplicate_containment_min_box_iou", type=float, default=0.25)
    parser.add_argument("--drop_duplicate_min_peer_frames", type=int, default=4)
    parser.add_argument("--drop_short_fragment_tracks", type=int, default=0)
    parser.add_argument("--drop_fragment_max_frames", type=int, default=30)
    parser.add_argument("--drop_fragment_min_peer_ratio", type=float, default=1.2)
    parser.add_argument("--drop_fragment_min_support_frames", type=int, default=6)
    parser.add_argument("--drop_fragment_min_mask_iou", type=float, default=0.20)
    parser.add_argument("--drop_fragment_min_containment", type=float, default=0.55)
    parser.add_argument("--drop_fragment_min_box_iou", type=float, default=0.35)
    parser.add_argument("--drop_edge_flicker_tracks", type=int, default=0)
    parser.add_argument("--drop_edge_flicker_labels", default="car")
    parser.add_argument("--drop_edge_flicker_max_frames", type=int, default=2)
    parser.add_argument("--drop_edge_flicker_max_span", type=int, default=30)
    parser.add_argument("--drop_edge_flicker_margin_ratio", type=float, default=0.30)
    parser.add_argument("--drop_edge_flicker_max_area_ratio", type=float, default=0.005)
    parser.add_argument("--drop_short_label_tracks", type=int, default=0)
    parser.add_argument("--drop_short_labels", default="")
    parser.add_argument("--drop_short_label_max_frames", type=int, default=3)
    parser.add_argument("--drop_short_label_max_span", type=int, default=30)
    parser.add_argument("--drop_short_label_max_area_ratio", type=float, default=0.01)
    parser.add_argument("--gap_fill_short_tracks", type=int, default=0)
    parser.add_argument("--gap_fill_short_labels", default="car")
    parser.add_argument("--gap_fill_short_max_frames", type=int, default=3)
    parser.add_argument("--gap_fill_anchor_min_frames", type=int, default=8)
    parser.add_argument("--gap_fill_max_neighbor_gap", type=int, default=8)
    parser.add_argument("--gap_fill_max_center_dist", type=float, default=1.25)
    parser.add_argument("--gap_fill_min_mean_iou", type=float, default=0.0)
    parser.add_argument("--gap_fill_short_max_area_ratio", type=float, default=0.02)
    parser.add_argument("--gap_fill_center_scale_floor", type=float, default=18.0)
    parser.add_argument("--adjacent_flicker_merge_tracks", type=int, default=0)
    parser.add_argument("--adjacent_flicker_labels", default="car")
    parser.add_argument("--adjacent_flicker_min_support", type=int, default=2)
    parser.add_argument("--adjacent_flicker_min_box_iou", type=float, default=0.10)
    parser.add_argument("--adjacent_flicker_max_center_dist", type=float, default=0.80)
    parser.add_argument("--adjacent_flicker_min_area_ratio", type=float, default=0.15)
    parser.add_argument("--adjacent_flicker_center_scale_floor", type=float, default=24.0)
    parser.add_argument("--adjacent_flicker_max_common_frames", type=int, default=3)
    parser.add_argument("--adjacent_flicker_common_min_mask_iou", type=float, default=0.45)
    parser.add_argument("--adjacent_flicker_common_min_containment", type=float, default=0.70)
    parser.add_argument("--adjacent_flicker_allow_single_strong", type=int, default=1)
    parser.add_argument("--adjacent_flicker_single_min_box_iou", type=float, default=0.08)
    parser.add_argument("--adjacent_flicker_single_max_center_dist", type=float, default=0.70)
    parser.add_argument("--adjacent_flicker_single_min_area_ratio", type=float, default=0.35)
    parser.add_argument(
        "--manual_merge_track_groups",
        default="",
        help="Audited semicolon-separated track index groups to merge, e.g. '37,38,39;47,48,50'. Defaults off.",
    )
    parser.add_argument("--manual_merge_require_no_common_frames", type=int, default=1)
    parser.add_argument("--proposal_drift_repair", type=int, default=0)
    parser.add_argument("--proposal_tracklets_pt", default="")
    parser.add_argument("--proposal_chunks_root", default="")
    parser.add_argument("--track_metadata_json", default="")
    parser.add_argument("--proposal_repair_labels", default="person")
    parser.add_argument("--proposal_repair_tracklet_ids", default="")
    parser.add_argument("--proposal_repair_min_conf", type=float, default=0.50)
    parser.add_argument("--proposal_repair_max_box_iou", type=float, default=0.25)
    parser.add_argument("--proposal_repair_min_area_ratio", type=float, default=0.0003)
    parser.add_argument("--proposal_repair_max_area_ratio", type=float, default=0.80)
    parser.add_argument("--proposal_repair_add_missing_frames", type=int, default=0)
    parser.add_argument("--proposal_miss_fallback", type=int, default=0)
    parser.add_argument("--proposal_miss_labels", default="person")
    parser.add_argument("--proposal_miss_tracklet_ids", default="")
    parser.add_argument("--proposal_miss_min_conf", type=float, default=0.60)
    parser.add_argument("--proposal_miss_max_coverage", type=float, default=0.25)
    parser.add_argument("--proposal_miss_min_frames", type=int, default=3)
    parser.add_argument("--proposal_miss_min_consecutive_frames", type=int, default=1)
    parser.add_argument("--proposal_miss_max_frame_gap", type=int, default=0)
    parser.add_argument("--proposal_miss_min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--proposal_miss_max_area_ratio", type=float, default=0.80)
    parser.add_argument("--postmerge_proposal_repair", type=int, default=0)
    parser.add_argument("--postmerge_repair_labels", default="car")
    parser.add_argument("--postmerge_repair_tracklet_ids", default="")
    parser.add_argument("--postmerge_repair_min_conf", type=float, default=0.60)
    parser.add_argument("--postmerge_repair_support_min_conf", type=float, default=0.35)
    parser.add_argument("--postmerge_repair_max_coverage", type=float, default=0.25)
    parser.add_argument("--postmerge_repair_min_area_ratio", type=float, default=0.0005)
    parser.add_argument("--postmerge_repair_max_area_ratio", type=float, default=0.80)
    parser.add_argument("--postmerge_repair_min_support_frames", type=int, default=2)
    parser.add_argument("--postmerge_repair_min_support_coverage", type=float, default=0.50)
    parser.add_argument("--postmerge_repair_min_support_box_iou", type=float, default=0.50)
    parser.add_argument("--contact_frames", default="0,100,200,300,400,500,600,700,800,900,1000,1100")
    return parser.parse_args()


def _frame_cache_manifest(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> Dict[str, Any]:
    return {
        "input_video": str(Path(args.input_video).resolve()),
        "processing_max_side": int(args.processing_max_side),
        "frames_limit": int(args.frames_limit),
        "frame_height": int(expected_h),
        "frame_width": int(expected_w),
        "num_frames": int(num_frames),
    }


def _load_frame_cache(
    cache_dir: Path,
    expected_manifest: Dict[str, Any],
) -> Optional[List[str]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            return None

    num_frames = int(expected_manifest["num_frames"])
    expected_h = int(expected_manifest["frame_height"])
    expected_w = int(expected_manifest["frame_width"])
    paths = [cache_dir / f"frame_{idx:06d}.jpg" for idx in range(num_frames)]
    if not all(path.is_file() for path in paths):
        return None
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None or first.shape[:2] != (expected_h, expected_w):
        return None
    return [str(path) for path in paths]


def _write_frame_cache(cache_dir: Path, image_paths: List[str], manifest: Dict[str, Any]) -> List[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stale_patterns = ("frame_*.jpg", "frame_*.jpeg", "frame_*.png")
    for pattern in stale_patterns:
        for path in cache_dir.glob(pattern):
            path.unlink(missing_ok=True)
    cached_paths: List[str] = []
    for idx, src in enumerate(image_paths):
        dst = cache_dir / f"frame_{idx:06d}.jpg"
        if Path(src).resolve() == dst.resolve():
            cached_paths.append(str(dst))
            continue
        shutil.copy2(src, dst)
        cached_paths.append(str(dst))
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return cached_paths


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> tuple[List[str], List[str]]:
    cache_dir_arg = str(getattr(args, "frame_cache_dir", "") or "").strip()
    expected_manifest = _frame_cache_manifest(args, expected_h, expected_w, num_frames)
    if cache_dir_arg:
        cache_dir = Path(cache_dir_arg)
        if not int(getattr(args, "frame_cache_refresh", 0)):
            cached = _load_frame_cache(cache_dir, expected_manifest)
            if cached is not None:
                print(f"Loaded {len(cached)} cached processing frames from {cache_dir}")
                return cached, []

    image_paths, temp_dir = collect_image_paths(args.input_video, 0, int(num_frames), 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if args.frames_limit and int(args.frames_limit) > 0:
        image_paths = image_paths[: int(args.frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < num_frames:
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[:num_frames]
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    if cache_dir_arg:
        cached_paths = _write_frame_cache(Path(cache_dir_arg), image_paths, expected_manifest)
        print(f"Wrote {len(cached_paths)} processing frames to cache {cache_dir_arg}")
        return cached_paths, temp_dirs
    return list(image_paths), temp_dirs


def _as_box(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4].copy()


def _as_mask_np(value: Any, H: int, W: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mask = np.asarray(value)
    if mask.shape != (H, W):
        return None
    return mask.astype(bool)


def _box_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (area_a + area_b - inter + 1e-6))


def _box_containment_small(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return float(inter / (min(area_a, area_b) + 1e-6))


def _center_dist_norm(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 999.0
    ac = np.asarray([(float(a[0]) + float(a[2])) * 0.5, (float(a[1]) + float(a[3])) * 0.5])
    bc = np.asarray([(float(b[0]) + float(b[2])) * 0.5, (float(b[1]) + float(b[3])) * 0.5])
    scale = max(np.sqrt(max(float(a[2]) - float(a[0]), 1.0) * max(float(a[3]) - float(a[1]), 1.0)), 1.0)
    return float(np.linalg.norm(ac - bc) / scale)


def _center_dist_norm_floor(a: Optional[np.ndarray], b: Optional[np.ndarray], scale_floor: float) -> float:
    if a is None or b is None:
        return 999.0
    ac = np.asarray([(float(a[0]) + float(a[2])) * 0.5, (float(a[1]) + float(a[3])) * 0.5])
    bc = np.asarray([(float(b[0]) + float(b[2])) * 0.5, (float(b[1]) + float(b[3])) * 0.5])
    scale = max(
        np.sqrt(max(float(a[2]) - float(a[0]), 1.0) * max(float(a[3]) - float(a[1]), 1.0)),
        float(scale_floor),
        1.0,
    )
    return float(np.linalg.norm(ac - bc) / scale)


def _box_velocity(track: Dict[str, Any], frames: List[int], at_end: bool) -> Optional[np.ndarray]:
    if len(frames) < 2:
        return None
    if at_end:
        f0, f1 = int(frames[-2]), int(frames[-1])
    else:
        f0, f1 = int(frames[0]), int(frames[1])
    if f1 == f0:
        return None
    b0 = _as_box(track.get("box_by_frame", {}).get(f0))
    b1 = _as_box(track.get("box_by_frame", {}).get(f1))
    if b0 is None or b1 is None:
        return None
    return (b1 - b0) / float(f1 - f0)


def _predict_box(track: Dict[str, Any], frame_from: int, frame_to: int, velocity: Optional[np.ndarray]) -> Optional[np.ndarray]:
    box = _as_box(track.get("box_by_frame", {}).get(int(frame_from)))
    if box is None:
        return None
    if velocity is None:
        return box
    return box + velocity * float(int(frame_to) - int(frame_from))


def _mask_iou(track_a: Dict[str, Any], track_b: Dict[str, Any], frame_idx: int, H: int, W: int) -> float:
    packed_a = track_a.get("mask_by_frame", {}).get(int(frame_idx))
    packed_b = track_b.get("mask_by_frame", {}).get(int(frame_idx))
    if packed_a is None or packed_b is None:
        return 0.0
    mask_a = _unpack_mask_np(np.asarray(packed_a, dtype=np.uint8), H, W).astype(bool)
    mask_b = _unpack_mask_np(np.asarray(packed_b, dtype=np.uint8), H, W).astype(bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum() / float(union))


def _mask_iou_and_containment(
    track_a: Dict[str, Any],
    track_b: Dict[str, Any],
    frame_idx: int,
    H: int,
    W: int,
) -> Tuple[float, float, int, int, int]:
    packed_a = track_a.get("mask_by_frame", {}).get(int(frame_idx))
    packed_b = track_b.get("mask_by_frame", {}).get(int(frame_idx))
    if packed_a is None or packed_b is None:
        return 0.0, 0.0, 0, 0, 0
    mask_a = _unpack_mask_np(np.asarray(packed_a, dtype=np.uint8), H, W).astype(bool)
    mask_b = _unpack_mask_np(np.asarray(packed_b, dtype=np.uint8), H, W).astype(bool)
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    if area_a <= 0 or area_b <= 0:
        return 0.0, 0.0, area_a, area_b, 0
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    mask_iou = 0.0 if union <= 0 else float(inter / float(union))
    containment = float(inter / float(min(area_a, area_b) + 1e-6))
    return mask_iou, containment, area_a, area_b, inter


def _frames(track: Dict[str, Any]) -> List[int]:
    return sorted(int(frame_idx) for frame_idx in track.get("mask_by_frame", {}).keys())


def _quality(track: Dict[str, Any], frame_idx: int) -> tuple[float, float]:
    score = float(track.get("q_by_frame", {}).get(int(frame_idx), 0.0))
    area = float(track.get("area_by_frame", {}).get(int(frame_idx), 0.0))
    return score, area


def _pair_decision(
    a: Dict[str, Any],
    b: Dict[str, Any],
    H: int,
    W: int,
    args: argparse.Namespace,
) -> tuple[bool, str, float]:
    if str(a.get("source_type")) != "thing_tracked" or str(b.get("source_type")) != "thing_tracked":
        return False, "non_thing", 0.0
    if str(a.get("L_sem", "")).lower() != str(b.get("L_sem", "")).lower():
        return False, "label_mismatch", 0.0
    fa = _frames(a)
    fb = _frames(b)
    if len(fa) < int(args.min_track_frames) or len(fb) < int(args.min_track_frames):
        return False, "too_short", 0.0
    overlap = sorted(set(fa).intersection(fb))
    if overlap:
        best = max(_mask_iou(a, b, frame_idx, H, W) for frame_idx in overlap)
        return best >= float(args.overlap_iou), "overlap_iou", float(best)

    if fa[-1] < fb[0]:
        gap = int(fb[0]) - int(fa[-1])
        left, right = a, b
        left_frame, right_frame = fa[-1], fb[0]
    elif fb[-1] < fa[0]:
        gap = int(fa[0]) - int(fb[-1])
        left, right = b, a
        left_frame, right_frame = fb[-1], fa[0]
    else:
        return False, "interleaved_no_overlap", 0.0
    if gap < 0 or gap > int(args.max_gap):
        return False, "gap_too_large", float(gap)

    box_iou = _box_iou(
        _as_box(left.get("box_by_frame", {}).get(int(left_frame))),
        _as_box(right.get("box_by_frame", {}).get(int(right_frame))),
    )
    center = _center_dist_norm(
        _as_box(left.get("box_by_frame", {}).get(int(left_frame))),
        _as_box(right.get("box_by_frame", {}).get(int(right_frame))),
    )
    ok = box_iou >= float(args.gap_box_iou) or center <= float(args.center_dist)
    score = max(float(box_iou), float(1.0 / (1.0 + center)))
    reason = f"gap_{gap}_box_iou_{box_iou:.3f}_center_{center:.3f}"
    return ok, reason, score


def _copy_frame(dst: Dict[str, Any], src: Dict[str, Any], frame_idx: int) -> None:
    frame_idx = int(frame_idx)
    dst["mask_by_frame"][frame_idx] = src["mask_by_frame"][frame_idx]
    dst["box_by_frame"][frame_idx] = src["box_by_frame"][frame_idx]
    dst["q_by_frame"][frame_idx] = src["q_by_frame"].get(frame_idx, 1.0)
    dst["area_by_frame"][frame_idx] = src["area_by_frame"].get(frame_idx, 0.0)


def _is_sam3_refined_track(track: Dict[str, Any]) -> bool:
    fields = (
        track.get("mask_source", ""),
        track.get("tracking_source", ""),
        track.get("sam3_status", ""),
    )
    return any("sam3" in str(value).lower() for value in fields)


def _merge_members(
    members: List[Dict[str, Any]],
    history_event: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    members = sorted(members, key=lambda t: (min(_frames(t) or [0]), -len(_frames(t))))
    base = members[0]
    provenance_keys = [
        "mask_source",
        "proposal_source",
        "tracking_source",
        "label_source",
        "semantic_resolver",
        "sam3_status",
    ]
    merged = {
        "mask_by_frame": {},
        "box_by_frame": {},
        "q_by_frame": {},
        "area_by_frame": {},
        "L_sem": base.get("L_sem"),
        "G_sem": int(base.get("G_sem", 0)),
        "W_sem": float(base.get("W_sem", 0.0)),
        "source_type": base.get("source_type"),
        "birth_frame": min(int(t.get("birth_frame", min(_frames(t) or [0]))) for t in members),
        "frame_height": int(base.get("frame_height", 0)),
        "frame_width": int(base.get("frame_width", 0)),
    }
    for key in provenance_keys:
        values = [str(track.get(key, "")) for track in members if str(track.get(key, ""))]
        uniq = sorted(set(values))
        if not uniq:
            continue
        merged[key] = uniq[0] if len(uniq) == 1 else "+".join(uniq)
    history: List[Dict[str, Any]] = []
    for track in members:
        raw_history = track.get("postprocess_history", [])
        if isinstance(raw_history, list):
            history.extend(item for item in raw_history if isinstance(item, dict))
    if history_event is not None:
        event = dict(history_event)
        event.setdefault("op", "merge_sparse_thing_tracks")
        event.setdefault("member_count", int(len(members)))
        event.setdefault("member_frame_ranges", [
            [int(frames[0]), int(frames[-1])] if (frames := _frames(track)) else []
            for track in members
        ])
        history.append(event)
    if history:
        merged["postprocess_history"] = history
    merged_frame_is_sam3: Dict[int, bool] = {}
    for track in members:
        track_is_sam3 = _is_sam3_refined_track(track)
        for frame_idx in _frames(track):
            if frame_idx not in merged["mask_by_frame"]:
                _copy_frame(merged, track, frame_idx)
                merged_frame_is_sam3[int(frame_idx)] = bool(track_is_sam3)
                continue
            current_is_sam3 = bool(merged_frame_is_sam3.get(int(frame_idx), False))
            if current_is_sam3 and not track_is_sam3:
                continue
            if track_is_sam3 and not current_is_sam3:
                _copy_frame(merged, track, frame_idx)
                merged_frame_is_sam3[int(frame_idx)] = True
                continue
            if _quality(track, frame_idx) > _quality(merged, frame_idx):
                _copy_frame(merged, track, frame_idx)
                merged_frame_is_sam3[int(frame_idx)] = bool(track_is_sam3)
    return merged


def _label_set(raw: str) -> set[str]:
    return {item.strip().lower() for item in str(raw or "").split(",") if item.strip()}


def _labels_match_or_vehicle_compatible(
    left_label: str,
    right_label: str,
    enabled: bool,
    vehicle_labels: set[str],
) -> bool:
    left = str(left_label or "").lower()
    right = str(right_label or "").lower()
    if left == right:
        return True
    return bool(enabled and left in vehicle_labels and right in vehicle_labels)


def _canonicalize_vehicle_labels(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.canonicalize_vehicle_labels):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "changed_tracks": 0,
        }
    labels = _label_set(args.canonical_vehicle_labels)
    output_label = str(args.canonical_vehicle_output_label or "car").strip() or "car"
    output: List[Dict[str, Any]] = []
    changes: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        label = str(track.get("L_sem", "")).lower()
        if str(track.get("source_type")) == "thing_tracked" and label in labels and label != output_label.lower():
            updated = dict(track)
            updated["L_sem"] = output_label
            output.append(updated)
            frames = _frames(track)
            changes.append(
                {
                    "track_index": int(idx),
                    "from_label": str(track.get("L_sem", "")),
                    "to_label": output_label,
                    "frames": int(len(frames)),
                    "range": [int(frames[0]), int(frames[-1])] if frames else [],
                }
            )
        else:
            output.append(track)
    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output)),
        "changed_tracks": int(len(changes)),
        "canonical_vehicle_labels": sorted(labels),
        "canonical_vehicle_output_label": output_label,
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output),
        "changes_preview": changes[:200],
    }
    return output, debug


def _load_proposals_by_id(chunks_root: Path) -> Dict[int, Dict[str, Any]]:
    proposals: Dict[int, Dict[str, Any]] = {}
    for path in sorted(chunks_root.glob("chunk_*/thing_proposals.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for rec in payload.get("proposals", []):
            proposals[int(rec["proposal_id"])] = rec
    return proposals


def _proposal_to_tracklet(tracklet_payload: Dict[str, Any]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for rec in tracklet_payload.get("tracklets", []):
        tracklet_id = int(rec.get("tracklet_id", -1))
        for proposal_id in rec.get("proposal_ids", []):
            mapping[int(proposal_id)] = tracklet_id
    return mapping


def _proposal_box(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    return _as_box(rec.get("box"))


def _proposal_mask(rec: Dict[str, Any], H: int, W: int) -> Optional[np.ndarray]:
    return _as_mask_np(rec.get("mask"), H, W)


def _proposal_area_ratio(rec: Dict[str, Any], mask: np.ndarray) -> float:
    value = rec.get("area_ratio")
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    return float(mask.sum() / float(mask.size + 1e-6))


def _repair_sam3_drift_with_proposals(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.proposal_drift_repair):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "repaired_frames": 0,
        }
    required = {
        "proposal_tracklets_pt": str(args.proposal_tracklets_pt or ""),
        "proposal_chunks_root": str(args.proposal_chunks_root or ""),
        "track_metadata_json": str(args.track_metadata_json or ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return tracks, {
            "enabled": True,
            "skipped": f"missing_args:{','.join(missing)}",
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "repaired_frames": 0,
        }

    proposal_tracklets_path = Path(args.proposal_tracklets_pt)
    proposal_chunks_root = Path(args.proposal_chunks_root)
    track_metadata_path = Path(args.track_metadata_json)
    if not proposal_tracklets_path.exists() or not proposal_chunks_root.exists() or not track_metadata_path.exists():
        return tracks, {
            "enabled": True,
            "skipped": "missing_files",
            "proposal_tracklets_pt": str(proposal_tracklets_path),
            "proposal_chunks_root": str(proposal_chunks_root),
            "track_metadata_json": str(track_metadata_path),
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "repaired_frames": 0,
        }

    metadata = json.loads(track_metadata_path.read_text(encoding="utf-8"))
    tracklet_payload = torch.load(proposal_tracklets_path, map_location="cpu", weights_only=False)
    tracklet_to_proposal_ids = {
        int(rec["tracklet_id"]): [int(pid) for pid in rec.get("proposal_ids", [])]
        for rec in tracklet_payload.get("tracklets", [])
    }
    proposals_by_id = _load_proposals_by_id(proposal_chunks_root)
    labels = _label_set(args.proposal_repair_labels)
    target_tracklet_ids = {int(item) for item in _label_set(args.proposal_repair_tracklet_ids)}

    repaired_frames = 0
    added_missing_frames = 0
    affected_tracks: set[int] = set()
    repairs: List[Dict[str, Any]] = []
    for track_index_raw, meta in metadata.items():
        track_index = int(track_index_raw)
        if track_index < 0 or track_index >= len(tracks):
            continue
        track = tracks[track_index]
        label = str(track.get("L_sem", "")).lower()
        if labels and label not in labels:
            continue
        if "thing" not in str(track.get("source_type", "")).lower():
            continue
        tracklet_id = int(meta.get("proposal_tracklet_id", -1))
        if target_tracklet_ids and tracklet_id not in target_tracklet_ids:
            continue
        proposal_ids = tracklet_to_proposal_ids.get(tracklet_id, [])
        if not proposal_ids:
            continue
        for proposal_id in proposal_ids:
            proposal = proposals_by_id.get(int(proposal_id))
            if proposal is None:
                continue
            frame_idx = int(proposal.get("frame_idx", -1))
            has_frame = frame_idx in track.get("mask_by_frame", {})
            if not has_frame and not int(args.proposal_repair_add_missing_frames):
                continue
            if str(proposal.get("label", "")).lower() != label:
                continue
            conf = float(proposal.get("confidence", 0.0))
            if conf < float(args.proposal_repair_min_conf):
                continue
            proposal_mask = _proposal_mask(proposal, H, W)
            if proposal_mask is None:
                continue
            area_ratio = _proposal_area_ratio(proposal, proposal_mask)
            if area_ratio < float(args.proposal_repair_min_area_ratio) or area_ratio > float(args.proposal_repair_max_area_ratio):
                continue
            current_box = _as_box(track.get("box_by_frame", {}).get(frame_idx))
            proposal_box = _proposal_box(proposal)
            if proposal_box is None:
                continue
            box_iou = _box_iou(current_box, proposal_box)
            if has_frame and box_iou > float(args.proposal_repair_max_box_iou):
                continue
            track["mask_by_frame"][frame_idx] = _pack_mask_np(proposal_mask)
            track["box_by_frame"][frame_idx] = torch.tensor(proposal_box, dtype=torch.float32)
            track["q_by_frame"][frame_idx] = conf
            track["area_by_frame"][frame_idx] = area_ratio
            repaired_frames += 1
            if not has_frame:
                added_missing_frames += 1
            affected_tracks.add(track_index)
            if len(repairs) < 300:
                repairs.append(
                    {
                        "action": "add_missing_frame" if not has_frame else "replace_drift_frame",
                        "track_index": int(track_index),
                        "proposal_tracklet_id": int(tracklet_id),
                        "proposal_id": int(proposal_id),
                        "frame_idx": int(frame_idx),
                        "label": label,
                        "confidence": float(conf),
                        "box_iou_before": float(box_iou),
                        "proposal_box": [float(x) for x in proposal_box.tolist()],
                        "current_box_before": [] if current_box is None else [float(x) for x in current_box.tolist()],
                    }
                )

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(tracks)),
        "repaired_tracks": int(len(affected_tracks)),
        "repaired_frames": int(repaired_frames),
        "added_missing_frames": int(added_missing_frames),
        "proposal_tracklets_pt": str(proposal_tracklets_path),
        "proposal_chunks_root": str(proposal_chunks_root),
        "track_metadata_json": str(track_metadata_path),
        "proposal_repair_labels": sorted(labels),
        "proposal_repair_tracklet_ids": sorted(target_tracklet_ids),
        "proposal_repair_min_conf": float(args.proposal_repair_min_conf),
        "proposal_repair_max_box_iou": float(args.proposal_repair_max_box_iou),
        "proposal_repair_min_area_ratio": float(args.proposal_repair_min_area_ratio),
        "proposal_repair_max_area_ratio": float(args.proposal_repair_max_area_ratio),
        "proposal_repair_add_missing_frames": int(args.proposal_repair_add_missing_frames),
        "repairs_preview": repairs,
    }
    return tracks, debug


def _person_union_by_frame(tracks: List[Dict[str, Any]], labels: set[str], H: int, W: int) -> Dict[int, np.ndarray]:
    unions: Dict[int, np.ndarray] = {}
    for track in tracks:
        if "thing" not in str(track.get("source_type", "")).lower():
            continue
        if str(track.get("L_sem", "")).lower() not in labels:
            continue
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            frame_idx = int(frame_idx)
            mask = _as_mask_np(_unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W), H, W)
            if mask is None:
                continue
            if frame_idx not in unions:
                unions[frame_idx] = np.zeros((H, W), dtype=bool)
            unions[frame_idx] |= mask
    return unions


def _filter_rows_by_temporal_consistency(
    rows: List[Dict[str, Any]],
    min_consecutive_frames: int,
    max_frame_gap: int,
) -> List[Dict[str, Any]]:
    if int(min_consecutive_frames) <= 1:
        return rows
    if not rows:
        return []

    ordered = sorted(rows, key=lambda row: int(row["frame_idx"]))
    max_gap = max(int(max_frame_gap), 0)
    kept: List[Dict[str, Any]] = []
    run: List[Dict[str, Any]] = [ordered[0]]
    for row in ordered[1:]:
        gap = int(row["frame_idx"]) - int(run[-1]["frame_idx"])
        if gap <= max_gap + 1:
            run.append(row)
        else:
            if len(run) >= int(min_consecutive_frames):
                kept.extend(run)
            run = [row]
    if len(run) >= int(min_consecutive_frames):
        kept.extend(run)
    return kept


def _add_missed_proposal_fallback_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.proposal_miss_fallback):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "added_tracks": 0,
            "added_frames": 0,
        }
    required = {
        "proposal_tracklets_pt": str(args.proposal_tracklets_pt or ""),
        "proposal_chunks_root": str(args.proposal_chunks_root or ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return tracks, {
            "enabled": True,
            "skipped": f"missing_args:{','.join(missing)}",
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "added_tracks": 0,
            "added_frames": 0,
        }

    proposal_tracklets_path = Path(args.proposal_tracklets_pt)
    proposal_chunks_root = Path(args.proposal_chunks_root)
    if not proposal_tracklets_path.exists() or not proposal_chunks_root.exists():
        return tracks, {
            "enabled": True,
            "skipped": "missing_files",
            "proposal_tracklets_pt": str(proposal_tracklets_path),
            "proposal_chunks_root": str(proposal_chunks_root),
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "added_tracks": 0,
            "added_frames": 0,
        }

    labels = _label_set(args.proposal_miss_labels)
    target_tracklet_ids = {int(item) for item in _label_set(args.proposal_miss_tracklet_ids)}
    tracklet_payload = torch.load(proposal_tracklets_path, map_location="cpu", weights_only=False)
    proposal_to_tracklet = _proposal_to_tracklet(tracklet_payload)
    proposals = list(_load_proposals_by_id(proposal_chunks_root).values())
    unions = _person_union_by_frame(tracks, labels, H, W)
    candidates: List[Dict[str, Any]] = []
    total_considered = 0
    for proposal in proposals:
        proposal_id = int(proposal.get("proposal_id", -1))
        tracklet_id = int(proposal_to_tracklet.get(proposal_id, -1))
        label = str(proposal.get("label", "")).lower()
        if labels and label not in labels:
            continue
        if target_tracklet_ids and tracklet_id not in target_tracklet_ids:
            continue
        conf = float(proposal.get("confidence", 0.0))
        if conf < float(args.proposal_miss_min_conf):
            continue
        proposal_mask = _proposal_mask(proposal, H, W)
        if proposal_mask is None:
            continue
        area_ratio = _proposal_area_ratio(proposal, proposal_mask)
        if area_ratio < float(args.proposal_miss_min_area_ratio) or area_ratio > float(args.proposal_miss_max_area_ratio):
            continue
        total_considered += 1
        frame_idx = int(proposal.get("frame_idx", -1))
        union = unions.get(frame_idx)
        coverage = 0.0
        if union is not None:
            area = int(proposal_mask.sum())
            coverage = 0.0 if area <= 0 else float(np.logical_and(proposal_mask, union).sum() / float(area))
        if coverage > float(args.proposal_miss_max_coverage):
            continue
        score = float(conf * area_ratio * (1.0 - coverage))
        candidates.append(
            {
                "score": score,
                "tracklet_id": tracklet_id,
                "proposal_id": proposal_id,
                "frame_idx": frame_idx,
                "label": label,
                "confidence": conf,
                "area_ratio": area_ratio,
                "coverage": coverage,
                "proposal": proposal,
                "proposal_mask": proposal_mask,
            }
        )

    candidates.sort(key=lambda row: (row["score"], row["confidence"], row["area_ratio"]), reverse=True)
    accepted_by_tracklet: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for row in candidates:
        frame_idx = int(row["frame_idx"])
        proposal_mask = row["proposal_mask"]
        union = unions.get(frame_idx)
        if union is not None:
            area = int(proposal_mask.sum())
            coverage = 0.0 if area <= 0 else float(np.logical_and(proposal_mask, union).sum() / float(area))
            if coverage > float(args.proposal_miss_max_coverage):
                continue
        tracklet_id = int(row["tracklet_id"])
        if tracklet_id < 0:
            continue
        prev = accepted_by_tracklet.setdefault(tracklet_id, {}).get(frame_idx)
        if prev is None or float(row["score"]) > float(prev["score"]):
            accepted_by_tracklet[tracklet_id][frame_idx] = row
        if frame_idx not in unions:
            unions[frame_idx] = np.zeros((H, W), dtype=bool)
        unions[frame_idx] |= proposal_mask

    output = list(tracks)
    added_tracks: List[Dict[str, Any]] = []
    rejected_tracklets: List[Dict[str, Any]] = []
    for tracklet_id, rows_by_frame in sorted(accepted_by_tracklet.items()):
        rows = [rows_by_frame[frame_idx] for frame_idx in sorted(rows_by_frame)]
        rows_before_temporal = rows
        rows = _filter_rows_by_temporal_consistency(
            rows,
            int(args.proposal_miss_min_consecutive_frames),
            int(args.proposal_miss_max_frame_gap),
        )
        if len(rows) < int(args.proposal_miss_min_frames):
            if len(rejected_tracklets) < 100:
                rejected_tracklets.append(
                    {
                        "proposal_tracklet_id": int(tracklet_id),
                        "candidate_frames": [int(row["frame_idx"]) for row in rows_before_temporal],
                        "kept_after_temporal": [int(row["frame_idx"]) for row in rows],
                        "candidate_frame_count": int(len(rows_before_temporal)),
                        "kept_frame_count": int(len(rows)),
                        "reason": "lt_min_frames_after_temporal_filter",
                    }
                )
            continue
        first = rows[0]
        fallback = {
            "mask_by_frame": {},
            "box_by_frame": {},
            "q_by_frame": {},
            "area_by_frame": {},
            "L_sem": first["label"],
            "G_sem": 0,
            "W_sem": float(np.mean([row["confidence"] for row in rows])),
            "source_type": "thing_proposal_fallback",
            "birth_frame": int(min(row["frame_idx"] for row in rows)),
            "frame_height": int(H),
            "frame_width": int(W),
            "proposal_fallback_tracklet_id": int(tracklet_id),
        }
        for row in rows:
            frame_idx = int(row["frame_idx"])
            proposal = row["proposal"]
            proposal_box = _proposal_box(proposal)
            if proposal_box is None:
                continue
            fallback["mask_by_frame"][frame_idx] = _pack_mask_np(row["proposal_mask"])
            fallback["box_by_frame"][frame_idx] = torch.tensor(proposal_box, dtype=torch.float32)
            fallback["q_by_frame"][frame_idx] = float(row["confidence"])
            fallback["area_by_frame"][frame_idx] = float(row["area_ratio"])
        if fallback["mask_by_frame"]:
            output.append(fallback)
            added_tracks.append(
                {
                    "proposal_tracklet_id": int(tracklet_id),
                    "label": first["label"],
                    "frames": sorted(int(frame_idx) for frame_idx in fallback["mask_by_frame"].keys()),
                    "frame_count": int(len(fallback["mask_by_frame"])),
                    "mean_confidence": float(fallback["W_sem"]),
                    "max_score": float(max(row["score"] for row in rows)),
                    "preview": [
                        {
                            "frame_idx": int(row["frame_idx"]),
                            "proposal_id": int(row["proposal_id"]),
                            "confidence": float(row["confidence"]),
                            "area_ratio": float(row["area_ratio"]),
                            "coverage_before": float(row["coverage"]),
                        }
                        for row in rows[:20]
                    ],
                }
            )

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output)),
        "added_tracks": int(len(added_tracks)),
        "added_frames": int(sum(row["frame_count"] for row in added_tracks)),
        "total_considered": int(total_considered),
        "candidate_frames": int(len(candidates)),
        "proposal_tracklets_pt": str(proposal_tracklets_path),
        "proposal_chunks_root": str(proposal_chunks_root),
        "proposal_miss_labels": sorted(labels),
        "proposal_miss_tracklet_ids": sorted(target_tracklet_ids),
        "proposal_miss_min_conf": float(args.proposal_miss_min_conf),
        "proposal_miss_max_coverage": float(args.proposal_miss_max_coverage),
        "proposal_miss_min_frames": int(args.proposal_miss_min_frames),
        "proposal_miss_min_consecutive_frames": int(args.proposal_miss_min_consecutive_frames),
        "proposal_miss_max_frame_gap": int(args.proposal_miss_max_frame_gap),
        "proposal_miss_min_area_ratio": float(args.proposal_miss_min_area_ratio),
        "proposal_miss_max_area_ratio": float(args.proposal_miss_max_area_ratio),
        "added_preview": added_tracks[:100],
        "rejected_preview": rejected_tracklets,
    }
    return output, debug


def _masked_hsv_hist(
    track: Dict[str, Any],
    track_index: int,
    frame_idx: int,
    image_paths: Optional[List[str]],
    image_cache: Dict[int, np.ndarray],
    hist_cache: Dict[Tuple[int, int], Optional[np.ndarray]],
    H: int,
    W: int,
    min_pixels: int,
) -> Optional[np.ndarray]:
    if image_paths is None:
        return None
    frame_idx = int(frame_idx)
    key = (int(track_index), frame_idx)
    if key in hist_cache:
        return hist_cache[key]
    packed = track.get("mask_by_frame", {}).get(frame_idx)
    if packed is None or frame_idx < 0 or frame_idx >= len(image_paths):
        hist_cache[key] = None
        return None
    if frame_idx not in image_cache:
        image = cv2.imread(str(image_paths[frame_idx]), cv2.IMREAD_COLOR)
        if image is None:
            hist_cache[key] = None
            return None
        image_cache[frame_idx] = image
    mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(np.uint8)
    if int(mask.sum()) < int(min_pixels):
        hist_cache[key] = None
        return None
    hsv = cv2.cvtColor(image_cache[frame_idx], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, [24, 16], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    hist_cache[key] = hist
    return hist


def _hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))


def _area_ratio_pair(a: Dict[str, Any], a_frame: int, b: Dict[str, Any], b_frame: int) -> float:
    aa = float(a.get("area_by_frame", {}).get(int(a_frame), 0.0))
    bb = float(b.get("area_by_frame", {}).get(int(b_frame), 0.0))
    if aa <= 0.0 or bb <= 0.0:
        return 0.0
    return float(min(aa, bb) / (max(aa, bb) + 1e-6))


def _reid_pair_decision(
    tracks: List[Dict[str, Any]],
    i: int,
    j: int,
    args: argparse.Namespace,
    H: int,
    W: int,
    image_paths: Optional[List[str]],
    image_cache: Dict[int, np.ndarray],
    hist_cache: Dict[Tuple[int, int], Optional[np.ndarray]],
) -> Tuple[bool, str, float]:
    a, b = tracks[i], tracks[j]
    if str(a.get("source_type")) != "thing_tracked" or str(b.get("source_type")) != "thing_tracked":
        return False, "non_thing", 0.0
    if str(a.get("L_sem", "")).lower() != str(b.get("L_sem", "")).lower():
        return False, "label_mismatch", 0.0
    fa, fb = _frames(a), _frames(b)
    if len(fa) < int(args.reid_min_frames) or len(fb) < int(args.reid_min_frames):
        return False, "too_short_for_reid", 0.0
    if fa[-1] < fb[0]:
        left, right = a, b
        left_idx, right_idx = i, j
        left_frame, right_frame = fa[-1], fb[0]
    elif fb[-1] < fa[0]:
        left, right = b, a
        left_idx, right_idx = j, i
        left_frame, right_frame = fb[-1], fa[0]
    else:
        return False, "overlap_or_interleaved", 0.0
    gap = int(right_frame) - int(left_frame)
    if gap <= 0 or gap > int(args.reid_max_gap):
        return False, "reid_gap_too_large", float(gap)

    left_box = _as_box(left.get("box_by_frame", {}).get(int(left_frame)))
    right_box = _as_box(right.get("box_by_frame", {}).get(int(right_frame)))
    box_iou = _box_iou(left_box, right_box)
    center = _center_dist_norm(left_box, right_box)
    area_ratio = _area_ratio_pair(left, left_frame, right, right_frame)
    geometry_ok = box_iou >= float(args.reid_min_box_iou) or center <= float(args.reid_max_center_dist)
    if not geometry_ok:
        return False, f"weak_reid_geometry_gap_{gap}_box_iou_{box_iou:.3f}_center_{center:.3f}", 0.0
    if area_ratio < float(args.reid_min_area_ratio):
        return False, f"area_ratio_{area_ratio:.3f}", float(area_ratio)

    hist_left = _masked_hsv_hist(
        left,
        left_idx,
        left_frame,
        image_paths,
        image_cache,
        hist_cache,
        H,
        W,
        int(args.reid_min_mask_pixels),
    )
    hist_right = _masked_hsv_hist(
        right,
        right_idx,
        right_frame,
        image_paths,
        image_cache,
        hist_cache,
        H,
        W,
        int(args.reid_min_mask_pixels),
    )
    hist_sim = _hist_similarity(hist_left, hist_right)
    if hist_sim < float(args.reid_min_hist_similarity):
        return False, f"hist_similarity_{hist_sim:.3f}", float(hist_sim)
    score = 0.55 * hist_sim + 0.25 * box_iou + 0.20 * (1.0 / (1.0 + center))
    reason = (
        f"reid_gap_{gap}_hist_{hist_sim:.3f}_box_iou_{box_iou:.3f}_"
        f"center_{center:.3f}_area_ratio_{area_ratio:.3f}"
    )
    return True, reason, float(score)


def _reid_merge_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
    image_paths: Optional[List[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.reid_merge_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }
    if image_paths is None:
        return tracks, {
            "enabled": True,
            "skipped": "missing_image_paths",
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }
    n = len(tracks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        return True

    image_cache: Dict[int, np.ndarray] = {}
    hist_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
    decisions: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ok, reason, score = _reid_pair_decision(tracks, i, j, args, H, W, image_paths, image_cache, hist_cache)
            if ok and union(i, j):
                decisions.append(
                    {
                        "from_index": int(j),
                        "to_index": int(i),
                        "label": str(tracks[i].get("L_sem")),
                        "reason": reason,
                        "score": float(score),
                    }
                )

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        output_tracks.append(_merge_members([tracks[idx] for idx in indices]))

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "merged_tracks": int(len(tracks) - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "merge_pairs": int(len(decisions)),
        "reid_max_gap": int(args.reid_max_gap),
        "reid_min_frames": int(args.reid_min_frames),
        "reid_min_hist_similarity": float(args.reid_min_hist_similarity),
        "reid_min_box_iou": float(args.reid_min_box_iou),
        "reid_max_center_dist": float(args.reid_max_center_dist),
        "reid_min_area_ratio": float(args.reid_min_area_ratio),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "decisions_preview": decisions[:200],
    }
    return output_tracks, debug


def _temporal_link_pair_decision(
    left: Dict[str, Any],
    right: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, str, float, Dict[str, Any]]:
    fl = _frames(left)
    fr = _frames(right)
    if not fl or not fr:
        return False, "empty_track", 0.0, {}
    if fl[-1] >= fr[0]:
        return False, "overlap_or_reverse", 0.0, {}

    gap = int(fr[0]) - int(fl[-1])
    if gap <= 0 or gap > int(args.temporal_link_max_gap):
        return False, "gap_too_large", float(gap), {"gap": int(gap)}
    left_label = str(left.get("L_sem", "")).lower()
    right_label = str(right.get("L_sem", "")).lower()
    cross_label = left_label != right_label

    left_box = _as_box(left.get("box_by_frame", {}).get(int(fl[-1])))
    right_box = _as_box(right.get("box_by_frame", {}).get(int(fr[0])))
    box_iou = _box_iou(left_box, right_box)
    center = _center_dist_norm_floor(left_box, right_box, float(args.temporal_link_center_scale_floor))
    area_ratio = _area_ratio_pair(left, int(fl[-1]), right, int(fr[0]))
    if area_ratio < float(args.temporal_link_min_area_ratio):
        return False, f"area_ratio_{area_ratio:.3f}", float(area_ratio), {
            "gap": int(gap),
            "box_iou": float(box_iou),
            "center": float(center),
            "area_ratio": float(area_ratio),
        }

    left_vel = _box_velocity(left, fl, at_end=True)
    right_vel = _box_velocity(right, fr, at_end=False)
    left_pred = _predict_box(left, int(fl[-1]), int(fr[0]), left_vel)
    right_back = _predict_box(right, int(fr[0]), int(fl[-1]), right_vel)
    pred_iou_lr = _box_iou(left_pred, right_box)
    pred_center_lr = _center_dist_norm_floor(left_pred, right_box, float(args.temporal_link_center_scale_floor))
    pred_iou_rl = _box_iou(left_box, right_back)
    pred_center_rl = _center_dist_norm_floor(left_box, right_back, float(args.temporal_link_center_scale_floor))
    pred_iou = max(pred_iou_lr, pred_iou_rl)
    pred_center = min(pred_center_lr, pred_center_rl)

    direct_ok = (
        box_iou >= float(args.temporal_link_min_box_iou)
        and center <= float(args.temporal_link_max_center_dist)
    )
    pred_ok = pred_center <= float(args.temporal_link_max_pred_center_dist)
    if gap > int(args.temporal_link_long_gap):
        long_gap_ok = (
            box_iou >= float(args.temporal_link_long_gap_min_box_iou)
            or center <= float(args.temporal_link_long_gap_max_center_dist)
            or (pred_iou >= float(args.temporal_link_long_gap_min_box_iou) and pred_center <= float(args.temporal_link_max_pred_center_dist))
        )
    else:
        long_gap_ok = True

    cross_label_ok = True
    if cross_label:
        cross_label_ok = (
            gap <= int(args.temporal_link_cross_label_max_gap)
            and area_ratio >= float(args.temporal_link_min_area_ratio)
            and (
                (
                    box_iou >= float(args.temporal_link_cross_label_min_box_iou)
                    and center <= float(args.temporal_link_cross_label_max_center_dist)
                )
                or (
                    pred_iou >= float(args.temporal_link_cross_label_min_box_iou)
                    and pred_center <= float(args.temporal_link_cross_label_max_center_dist)
                )
            )
        )
    ok = bool((direct_ok or pred_ok) and long_gap_ok)
    ok = bool(ok and cross_label_ok)
    info = {
        "gap": int(gap),
        "left_label": left_label,
        "right_label": right_label,
        "cross_label": bool(cross_label),
        "box_iou": float(box_iou),
        "center": float(center),
        "area_ratio": float(area_ratio),
        "pred_iou": float(pred_iou),
        "pred_center": float(pred_center),
        "left_frames": int(len(fl)),
        "right_frames": int(len(fr)),
        "left_range": [int(fl[0]), int(fl[-1])],
        "right_range": [int(fr[0]), int(fr[-1])],
    }
    if not ok:
        return False, (
            f"weak_temporal_gap_{gap}_labels_{left_label}->{right_label}_box_iou_{box_iou:.3f}_center_{center:.3f}_"
            f"pred_iou_{pred_iou:.3f}_pred_center_{pred_center:.3f}"
        ), 0.0, info

    score = (
        0.40 * float(box_iou)
        + 0.20 * float(pred_iou)
        + 0.25 * (1.0 / (1.0 + float(center)))
        + 0.20 * (1.0 / (1.0 + float(pred_center)))
        + 0.15 * float(area_ratio)
        - 0.012 * float(gap)
    )
    reason = (
        f"temporal_gap_{gap}_labels_{left_label}->{right_label}_box_iou_{box_iou:.3f}_center_{center:.3f}_"
        f"area_{area_ratio:.3f}_pred_iou_{pred_iou:.3f}_pred_center_{pred_center:.3f}"
    )
    return True, reason, float(score), info


def _temporal_link_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.temporal_link_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }

    labels = _label_set(args.temporal_link_labels)
    vehicle_labels = _label_set(args.temporal_link_vehicle_labels)
    n = len(tracks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        return True

    candidates: List[Dict[str, Any]] = []
    rejected_preview: List[Dict[str, Any]] = []
    for i, left in enumerate(tracks):
        if str(left.get("source_type")) != "thing_tracked":
            continue
        label = str(left.get("L_sem", "")).lower()
        if labels and label not in labels:
            continue
        fl = _frames(left)
        if not fl:
            continue
        for j, right in enumerate(tracks):
            if i == j or str(right.get("source_type")) != "thing_tracked":
                continue
            right_label = str(right.get("L_sem", "")).lower()
            same_label = right_label == label
            vehicle_compatible = (
                bool(int(args.temporal_link_vehicle_compatible))
                and label in vehicle_labels
                and right_label in vehicle_labels
            )
            if not same_label and not vehicle_compatible:
                continue
            if labels and right_label not in labels:
                continue
            fr = _frames(right)
            if not fr or fl[-1] >= fr[0]:
                continue
            gap = int(fr[0]) - int(fl[-1])
            if gap <= 0 or gap > int(args.temporal_link_max_gap):
                continue
            ok, reason, score, info = _temporal_link_pair_decision(left, right, args)
            row = {
                "left_index": int(i),
                "right_index": int(j),
                "label": label,
                "reason": reason,
                "score": float(score),
                **info,
            }
            if ok and float(score) >= float(args.temporal_link_min_score):
                candidates.append(row)
            elif len(rejected_preview) < 100:
                if ok:
                    row["reason"] = f"score_below_min_{score:.3f}"
                rejected_preview.append(row)

    candidates.sort(key=lambda row: row["score"], reverse=True)
    used_successor: set[int] = set()
    used_predecessor: set[int] = set()
    decisions: List[Dict[str, Any]] = []
    for row in candidates:
        left_idx = int(row["left_index"])
        right_idx = int(row["right_index"])
        if left_idx in used_successor or right_idx in used_predecessor:
            continue
        if find(left_idx) == find(right_idx):
            continue
        if not union(left_idx, right_idx):
            continue
        used_successor.add(left_idx)
        used_predecessor.add(right_idx)
        decisions.append(row)

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        output_tracks.append(_merge_members([tracks[idx] for idx in indices]))

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "merged_tracks": int(len(tracks) - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "candidate_pairs": int(len(candidates)),
        "merge_pairs": int(len(decisions)),
        "temporal_link_labels": sorted(labels),
        "temporal_link_vehicle_compatible": bool(int(args.temporal_link_vehicle_compatible)),
        "temporal_link_vehicle_labels": sorted(vehicle_labels),
        "temporal_link_min_score": float(args.temporal_link_min_score),
        "temporal_link_cross_label_max_gap": int(args.temporal_link_cross_label_max_gap),
        "temporal_link_cross_label_min_box_iou": float(args.temporal_link_cross_label_min_box_iou),
        "temporal_link_cross_label_max_center_dist": float(args.temporal_link_cross_label_max_center_dist),
        "temporal_link_max_gap": int(args.temporal_link_max_gap),
        "temporal_link_min_box_iou": float(args.temporal_link_min_box_iou),
        "temporal_link_min_area_ratio": float(args.temporal_link_min_area_ratio),
        "temporal_link_max_center_dist": float(args.temporal_link_max_center_dist),
        "temporal_link_max_pred_center_dist": float(args.temporal_link_max_pred_center_dist),
        "temporal_link_center_scale_floor": float(args.temporal_link_center_scale_floor),
        "temporal_link_long_gap": int(args.temporal_link_long_gap),
        "temporal_link_long_gap_min_box_iou": float(args.temporal_link_long_gap_min_box_iou),
        "temporal_link_long_gap_max_center_dist": float(args.temporal_link_long_gap_max_center_dist),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output_tracks),
        "decisions_preview": decisions[:300],
        "rejected_preview": rejected_preview,
    }
    return output_tracks, debug


def _temporal_link_tracks_multi(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.temporal_link_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
            "passes_run": 0,
        }

    current = tracks
    pass_debugs: List[Dict[str, Any]] = []
    max_passes = max(1, int(args.temporal_link_passes))
    for pass_idx in range(max_passes):
        current, debug = _temporal_link_tracks(current, args)
        debug["pass_index"] = int(pass_idx + 1)
        pass_debugs.append(debug)
        if int(debug.get("merged_tracks", 0)) <= 0:
            break

    merged_tracks = int(len(tracks) - len(current))
    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(current)),
        "merged_tracks": merged_tracks,
        "passes_requested": int(max_passes),
        "passes_run": int(len(pass_debugs)),
        "merge_pairs": int(sum(int(row.get("merge_pairs", 0)) for row in pass_debugs)),
        "candidate_pairs": int(sum(int(row.get("candidate_pairs", 0)) for row in pass_debugs)),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(current),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(current),
        "pass_debugs": pass_debugs,
    }
    if pass_debugs:
        first = pass_debugs[0]
        for key in (
            "temporal_link_labels",
            "temporal_link_vehicle_compatible",
            "temporal_link_vehicle_labels",
            "temporal_link_min_score",
            "temporal_link_cross_label_max_gap",
            "temporal_link_cross_label_min_box_iou",
            "temporal_link_cross_label_max_center_dist",
            "temporal_link_max_gap",
            "temporal_link_min_box_iou",
            "temporal_link_min_area_ratio",
            "temporal_link_max_center_dist",
            "temporal_link_max_pred_center_dist",
            "temporal_link_center_scale_floor",
            "temporal_link_long_gap",
            "temporal_link_long_gap_min_box_iou",
            "temporal_link_long_gap_max_center_dist",
        ):
            if key in first:
                debug[key] = first[key]
        debug["decisions_preview"] = [
            {**row, "pass_index": int(pass_debug.get("pass_index", 0))}
            for pass_debug in pass_debugs
            for row in pass_debug.get("decisions_preview", [])[:120]
        ][:300]
    return current, debug


def _overlap_support_pair_decision(
    a: Dict[str, Any],
    b: Dict[str, Any],
    H: int,
    W: int,
    args: argparse.Namespace,
) -> Tuple[bool, str, float, Dict[str, Any]]:
    if str(a.get("source_type")) != "thing_tracked" or str(b.get("source_type")) != "thing_tracked":
        return False, "non_thing", 0.0, {}
    left_label = str(a.get("L_sem", "")).lower()
    right_label = str(b.get("L_sem", "")).lower()
    vehicle_labels = _label_set(args.overlap_support_vehicle_labels)
    labels_ok = _labels_match_or_vehicle_compatible(
        left_label,
        right_label,
        bool(int(args.overlap_support_vehicle_compatible)),
        vehicle_labels,
    )
    if not labels_ok:
        return False, "label_mismatch", 0.0, {}

    common = sorted(set(_frames(a)).intersection(_frames(b)))
    if len(common) < int(args.overlap_support_min_common_frames):
        return False, "too_few_common_frames", float(len(common)), {"common_frames": int(len(common))}

    mask_ious: List[float] = []
    containments: List[float] = []
    duplicate_frames: List[int] = []
    conflict_frames: List[int] = []
    for frame_idx in common:
        packed_a = a.get("mask_by_frame", {}).get(int(frame_idx))
        packed_b = b.get("mask_by_frame", {}).get(int(frame_idx))
        if packed_a is None or packed_b is None:
            continue
        mask_a = _unpack_mask_np(np.asarray(packed_a, dtype=np.uint8), H, W).astype(bool)
        mask_b = _unpack_mask_np(np.asarray(packed_b, dtype=np.uint8), H, W).astype(bool)
        area_a = int(mask_a.sum())
        area_b = int(mask_b.sum())
        if area_a <= 0 or area_b <= 0:
            mask_ious.append(0.0)
            containments.append(0.0)
            continue
        inter = int(np.logical_and(mask_a, mask_b).sum())
        union = int(np.logical_or(mask_a, mask_b).sum())
        mask_iou = 0.0 if union <= 0 else float(inter / float(union))
        containment = float(inter / float(min(area_a, area_b) + 1e-6))
        mask_ious.append(mask_iou)
        containments.append(containment)
        if mask_iou >= float(args.overlap_support_min_mask_iou):
            duplicate_frames.append(int(frame_idx))
        if (
            mask_iou < float(args.overlap_support_conflict_mask_iou)
            and containment < float(args.overlap_support_conflict_containment)
        ):
            conflict_frames.append(int(frame_idx))

    if not mask_ious:
        return False, "no_masks", 0.0, {"common_frames": int(len(common))}

    common_mask_frames = int(len(mask_ious))
    duplicate_ratio = float(len(duplicate_frames) / max(1, common_mask_frames))
    conflict_ratio = float(len(conflict_frames) / max(1, common_mask_frames))
    mean_mask_iou = float(np.mean(mask_ious))
    p75_mask_iou = float(np.percentile(mask_ious, 75))
    max_mask_iou = float(np.max(mask_ious))
    mean_containment = float(np.mean(containments))
    max_conflict_frames = int(args.overlap_support_max_conflict_frames)
    stats = {
        "common_frames": int(len(common)),
        "common_mask_frames": common_mask_frames,
        "duplicate_frames": int(len(duplicate_frames)),
        "conflict_frames": int(len(conflict_frames)),
        "left_label": left_label,
        "right_label": right_label,
        "cross_label": bool(left_label != right_label),
        "duplicate_ratio": duplicate_ratio,
        "conflict_ratio": conflict_ratio,
        "mean_mask_iou": mean_mask_iou,
        "p75_mask_iou": p75_mask_iou,
        "max_mask_iou": max_mask_iou,
        "mean_containment": mean_containment,
        "duplicate_frame_preview": duplicate_frames[:20],
        "conflict_frame_preview": conflict_frames[:20],
    }
    ok = (
        len(duplicate_frames) >= int(args.overlap_support_min_duplicate_frames)
        and duplicate_ratio >= float(args.overlap_support_min_duplicate_ratio)
        and mean_mask_iou >= float(args.overlap_support_min_mean_mask_iou)
        and p75_mask_iou >= float(args.overlap_support_min_p75_mask_iou)
        and mean_containment >= float(args.overlap_support_min_mean_containment)
        and (max_conflict_frames < 0 or len(conflict_frames) <= max_conflict_frames)
        and conflict_ratio <= float(args.overlap_support_max_conflict_ratio)
    )
    reason = (
        f"overlap_support_common_{len(common)}_dup_{len(duplicate_frames)}_"
        f"conflict_{len(conflict_frames)}_"
        f"mean_iou_{mean_mask_iou:.3f}_p75_iou_{p75_mask_iou:.3f}_"
        f"mean_containment_{mean_containment:.3f}_"
        f"dup_ratio_{duplicate_ratio:.3f}_conflict_ratio_{conflict_ratio:.3f}"
    )
    score = 0.45 * mean_mask_iou + 0.35 * p75_mask_iou + 0.20 * mean_containment
    return ok, reason, float(score), stats


def _overlap_support_merge_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.overlap_support_merge_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }

    n = len(tracks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        return True

    decisions: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ok, reason, score, stats = _overlap_support_pair_decision(tracks[i], tracks[j], H, W, args)
            if ok and union(i, j):
                decisions.append(
                    {
                        "from_index": int(j),
                        "to_index": int(i),
                        "label": str(tracks[i].get("L_sem")),
                        "reason": reason,
                        "score": float(score),
                        **stats,
                    }
                )
            elif stats and len(rejected) < 300:
                rejected.append(
                    {
                        "left_index": int(i),
                        "right_index": int(j),
                        "left_label": str(tracks[i].get("L_sem")),
                        "right_label": str(tracks[j].get("L_sem")),
                        "reason": reason,
                        "score": float(score),
                        **stats,
                    }
                )

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        output_tracks.append(_merge_members([tracks[idx] for idx in indices]))

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "merged_tracks": int(len(tracks) - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "merge_pairs": int(len(decisions)),
        "rejected_pairs_preview": rejected[:200],
        "overlap_support_min_common_frames": int(args.overlap_support_min_common_frames),
        "overlap_support_min_duplicate_frames": int(args.overlap_support_min_duplicate_frames),
        "overlap_support_min_mask_iou": float(args.overlap_support_min_mask_iou),
        "overlap_support_min_mean_mask_iou": float(args.overlap_support_min_mean_mask_iou),
        "overlap_support_min_p75_mask_iou": float(args.overlap_support_min_p75_mask_iou),
        "overlap_support_min_mean_containment": float(args.overlap_support_min_mean_containment),
        "overlap_support_min_duplicate_ratio": float(args.overlap_support_min_duplicate_ratio),
        "overlap_support_conflict_mask_iou": float(args.overlap_support_conflict_mask_iou),
        "overlap_support_conflict_containment": float(args.overlap_support_conflict_containment),
        "overlap_support_max_conflict_frames": int(args.overlap_support_max_conflict_frames),
        "overlap_support_max_conflict_ratio": float(args.overlap_support_max_conflict_ratio),
        "overlap_support_vehicle_compatible": bool(int(args.overlap_support_vehicle_compatible)),
        "overlap_support_vehicle_labels": sorted(_label_set(args.overlap_support_vehicle_labels)),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "decisions_preview": decisions[:200],
    }
    return output_tracks, debug


def _counter_by_label(tracks: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for track in tracks:
        label = str(track.get("L_sem", ""))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _short_by_label(tracks: List[Dict[str, Any]], max_len: int = 3) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for track in tracks:
        if str(track.get("source_type")) != "thing_tracked":
            continue
        if len(_frames(track)) <= max_len:
            label = str(track.get("L_sem", ""))
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _best_duplicate_peer(
    tracks: List[Dict[str, Any]],
    index: int,
    frame_idx: int,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Optional[Dict[str, Any]]:
    track = tracks[index]
    label = str(track.get("L_sem", "")).lower()
    frame_idx = int(frame_idx)
    own_box = _as_box(track.get("box_by_frame", {}).get(frame_idx))
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for peer_idx, peer in enumerate(tracks):
        if peer_idx == index:
            continue
        if str(peer.get("source_type")) != "thing_tracked":
            continue
        if str(peer.get("L_sem", "")).lower() != label:
            continue
        if frame_idx not in peer.get("mask_by_frame", {}):
            continue
        if len(_frames(peer)) < int(args.drop_duplicate_min_peer_frames) and len(_frames(peer)) <= len(_frames(track)):
            continue
        peer_box = _as_box(peer.get("box_by_frame", {}).get(frame_idx))
        box_iou = _box_iou(own_box, peer_box)
        containment = _box_containment_small(own_box, peer_box)
        mask_iou = 0.0
        if box_iou > 0.05 or containment > 0.3:
            mask_iou = _mask_iou(track, peer, frame_idx, H, W)
        ok = (
            mask_iou >= float(args.drop_duplicate_mask_iou)
            or box_iou >= float(args.drop_duplicate_box_iou)
            or (
                containment >= float(args.drop_duplicate_containment)
                and box_iou >= float(args.drop_duplicate_containment_min_box_iou)
            )
        )
        score = max(mask_iou, box_iou, containment * 0.75)
        if ok and score > best_score:
            best_score = score
            best = {
                "peer_index": int(peer_idx),
                "frame_idx": int(frame_idx),
                "mask_iou": float(mask_iou),
                "box_iou": float(box_iou),
                "containment": float(containment),
                "peer_frames": int(len(_frames(peer))),
            }
    return best


def _best_trim_duplicate_peer(
    tracks: List[Dict[str, Any]],
    index: int,
    frame_idx: int,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Optional[Dict[str, Any]]:
    track = tracks[index]
    label = str(track.get("L_sem", "")).lower()
    own_frames = len(_frames(track))
    own_box = _as_box(track.get("box_by_frame", {}).get(int(frame_idx)))
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for peer_idx, peer in enumerate(tracks):
        if peer_idx == index:
            continue
        if str(peer.get("source_type")) != "thing_tracked":
            continue
        if str(peer.get("L_sem", "")).lower() != label:
            continue
        if int(frame_idx) not in peer.get("mask_by_frame", {}):
            continue
        peer_frames = len(_frames(peer))
        if peer_frames < int(args.trim_duplicate_min_peer_frames):
            continue
        if peer_frames < max(1.0, float(args.trim_duplicate_min_peer_ratio) * float(max(own_frames, 1))):
            continue
        peer_box = _as_box(peer.get("box_by_frame", {}).get(int(frame_idx)))
        box_iou = _box_iou(own_box, peer_box)
        if box_iou < 0.05:
            continue
        mask_iou = _mask_iou(track, peer, int(frame_idx), H, W)
        ok = mask_iou >= float(args.trim_duplicate_mask_iou) or box_iou >= float(args.trim_duplicate_box_iou)
        score = max(mask_iou, box_iou)
        if ok and score > best_score:
            best_score = score
            best = {
                "peer_index": int(peer_idx),
                "frame_idx": int(frame_idx),
                "mask_iou": float(mask_iou),
                "box_iou": float(box_iou),
                "peer_frames": int(peer_frames),
            }
    return best


def _copy_without_frames(track: Dict[str, Any], drop_frames: set[int]) -> Optional[Dict[str, Any]]:
    kept_frames = [frame_idx for frame_idx in _frames(track) if int(frame_idx) not in drop_frames]
    if not kept_frames:
        return None
    out = dict(track)
    for key in ("mask_by_frame", "box_by_frame", "q_by_frame", "area_by_frame"):
        values = track.get(key, {})
        if isinstance(values, dict):
            out[key] = {int(frame_idx): values[frame_idx] for frame_idx in kept_frames if frame_idx in values}
    out["birth_frame"] = int(min(kept_frames))
    return out


def _trim_duplicate_frames(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.trim_duplicate_frames):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "trimmed_frames": 0,
            "dropped_empty_tracks": 0,
        }

    output: List[Dict[str, Any]] = []
    trims: List[Dict[str, Any]] = []
    dropped_empty = 0
    for idx, track in enumerate(tracks):
        frames = _frames(track)
        if str(track.get("source_type")) != "thing_tracked" or len(frames) > int(args.trim_duplicate_max_track_frames):
            output.append(track)
            continue

        drop_frames: set[int] = set()
        matches: List[Dict[str, Any]] = []
        for frame_idx in frames:
            match = _best_trim_duplicate_peer(tracks, idx, int(frame_idx), args, H, W)
            if match is None:
                continue
            drop_frames.add(int(frame_idx))
            matches.append(match)

        if not drop_frames:
            output.append(track)
            continue

        trimmed = _copy_without_frames(track, drop_frames)
        area_values = list(track.get("area_by_frame", {}).values())
        trims.append(
            {
                "track_index": int(idx),
                "label": str(track.get("L_sem")),
                "frames_before": frames,
                "frames_trimmed": sorted(int(frame_idx) for frame_idx in drop_frames),
                "frames_after": [] if trimmed is None else _frames(trimmed),
                "max_area_ratio": float(max(area_values) if area_values else 0.0),
                "matches": matches,
            }
        )
        if trimmed is None:
            dropped_empty += 1
        else:
            output.append(trimmed)

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output)),
        "affected_tracks": int(len(trims)),
        "trimmed_frames": int(sum(len(row["frames_trimmed"]) for row in trims)),
        "dropped_empty_tracks": int(dropped_empty),
        "trim_duplicate_max_track_frames": int(args.trim_duplicate_max_track_frames),
        "trim_duplicate_mask_iou": float(args.trim_duplicate_mask_iou),
        "trim_duplicate_box_iou": float(args.trim_duplicate_box_iou),
        "trim_duplicate_min_peer_frames": int(args.trim_duplicate_min_peer_frames),
        "trim_duplicate_min_peer_ratio": float(args.trim_duplicate_min_peer_ratio),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output),
        "trims_preview": trims[:200],
    }
    return output, debug


def _contained_mask_match(
    tracks: List[Dict[str, Any]],
    left_idx: int,
    right_idx: int,
    frame_idx: int,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Optional[Dict[str, Any]]:
    left = tracks[left_idx]
    right = tracks[right_idx]
    if str(left.get("source_type")) != "thing_tracked" or str(right.get("source_type")) != "thing_tracked":
        return None
    if str(left.get("L_sem", "")).lower() != str(right.get("L_sem", "")).lower():
        return None
    if int(frame_idx) not in left.get("mask_by_frame", {}) or int(frame_idx) not in right.get("mask_by_frame", {}):
        return None
    if len(_frames(left)) < int(args.trim_contained_min_peer_frames) and len(_frames(right)) < int(args.trim_contained_min_peer_frames):
        return None

    mask_left = _unpack_mask_np(
        np.asarray(left["mask_by_frame"][int(frame_idx)], dtype=np.uint8),
        H,
        W,
    ).astype(bool)
    mask_right = _unpack_mask_np(
        np.asarray(right["mask_by_frame"][int(frame_idx)], dtype=np.uint8),
        H,
        W,
    ).astype(bool)
    area_left = int(mask_left.sum())
    area_right = int(mask_right.sum())
    if area_left <= 0 or area_right <= 0:
        return None
    inter = int(np.logical_and(mask_left, mask_right).sum())
    small_area = min(area_left, area_right)
    containment = float(inter / float(small_area + 1e-6))
    if containment < float(args.trim_contained_min_containment):
        return None

    if area_left <= area_right:
        remove_idx, peer_idx = left_idx, right_idx
        remove_area_ratio = float(left.get("area_by_frame", {}).get(int(frame_idx), 0.0))
    else:
        remove_idx, peer_idx = right_idx, left_idx
        remove_area_ratio = float(right.get("area_by_frame", {}).get(int(frame_idx), 0.0))
    if remove_area_ratio > float(args.trim_contained_max_area_ratio):
        return None
    if len(_frames(tracks[peer_idx])) < int(args.trim_contained_min_peer_frames):
        return None

    return {
        "remove_index": int(remove_idx),
        "peer_index": int(peer_idx),
        "frame_idx": int(frame_idx),
        "label": str(tracks[remove_idx].get("L_sem")),
        "containment": float(containment),
        "remove_area_ratio": float(remove_area_ratio),
        "remove_frames": int(len(_frames(tracks[remove_idx]))),
        "peer_frames": int(len(_frames(tracks[peer_idx]))),
    }


def _trim_contained_duplicate_frames(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.trim_contained_duplicate_frames):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "trimmed_frames": 0,
            "dropped_empty_tracks": 0,
        }

    by_frame_label: Dict[Tuple[int, str], List[int]] = {}
    for idx, track in enumerate(tracks):
        if str(track.get("source_type")) != "thing_tracked":
            continue
        label = str(track.get("L_sem", "")).lower()
        for frame_idx in _frames(track):
            by_frame_label.setdefault((int(frame_idx), label), []).append(idx)

    drop_by_track: Dict[int, set[int]] = {}
    events: List[Dict[str, Any]] = []
    for (frame_idx, _label), indices in sorted(by_frame_label.items()):
        if len(indices) < 2:
            continue
        for pos, left_idx in enumerate(indices):
            for right_idx in indices[pos + 1 :]:
                match = _contained_mask_match(tracks, left_idx, right_idx, int(frame_idx), args, H, W)
                if match is None:
                    continue
                remove_idx = int(match["remove_index"])
                if int(frame_idx) in drop_by_track.get(remove_idx, set()):
                    continue
                drop_by_track.setdefault(remove_idx, set()).add(int(frame_idx))
                events.append(match)

    if not drop_by_track:
        return tracks, {
            "enabled": True,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "affected_tracks": 0,
            "trimmed_frames": 0,
            "dropped_empty_tracks": 0,
            "trim_contained_min_containment": float(args.trim_contained_min_containment),
            "trim_contained_max_area_ratio": float(args.trim_contained_max_area_ratio),
            "trim_contained_min_peer_frames": int(args.trim_contained_min_peer_frames),
            "label_counts_before": _counter_by_label(tracks),
            "label_counts_after": _counter_by_label(tracks),
            "short_thing_tracks_le3_before": _short_by_label(tracks),
            "short_thing_tracks_le3_after": _short_by_label(tracks),
            "trims_preview": [],
        }

    output: List[Dict[str, Any]] = []
    trims: List[Dict[str, Any]] = []
    dropped_empty = 0
    events_by_track: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        events_by_track.setdefault(int(event["remove_index"]), []).append(event)

    for idx, track in enumerate(tracks):
        drop_frames = drop_by_track.get(idx, set())
        if not drop_frames:
            output.append(track)
            continue
        trimmed = _copy_without_frames(track, drop_frames)
        area_values = list(track.get("area_by_frame", {}).values())
        trims.append(
            {
                "track_index": int(idx),
                "label": str(track.get("L_sem")),
                "frames_before": _frames(track),
                "frames_trimmed": sorted(int(frame_idx) for frame_idx in drop_frames),
                "frames_after": [] if trimmed is None else _frames(trimmed),
                "max_area_ratio": float(max(area_values) if area_values else 0.0),
                "matches": events_by_track.get(idx, [])[:50],
            }
        )
        if trimmed is None:
            dropped_empty += 1
        else:
            output.append(trimmed)

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output)),
        "affected_tracks": int(len(trims)),
        "trimmed_frames": int(sum(len(row["frames_trimmed"]) for row in trims)),
        "dropped_empty_tracks": int(dropped_empty),
        "trim_contained_min_containment": float(args.trim_contained_min_containment),
        "trim_contained_max_area_ratio": float(args.trim_contained_max_area_ratio),
        "trim_contained_min_peer_frames": int(args.trim_contained_min_peer_frames),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output),
        "trims_preview": trims[:200],
    }
    return output, debug


def _endpoint_distance(frames: List[int], frame_idx: int) -> int:
    if not frames:
        return 999999
    try:
        pos = frames.index(int(frame_idx))
    except ValueError:
        return 999999
    return int(min(pos, len(frames) - 1 - pos))


def _endpoint_duplicate_match(
    tracks: List[Dict[str, Any]],
    left_idx: int,
    right_idx: int,
    frame_idx: int,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Optional[Dict[str, Any]]:
    left = tracks[left_idx]
    right = tracks[right_idx]
    if str(left.get("source_type")) != "thing_tracked" or str(right.get("source_type")) != "thing_tracked":
        return None
    if str(left.get("L_sem", "")).lower() != str(right.get("L_sem", "")).lower():
        return None
    frame_idx = int(frame_idx)
    if frame_idx not in left.get("mask_by_frame", {}) or frame_idx not in right.get("mask_by_frame", {}):
        return None

    mask_left = _unpack_mask_np(np.asarray(left["mask_by_frame"][frame_idx], dtype=np.uint8), H, W).astype(bool)
    mask_right = _unpack_mask_np(np.asarray(right["mask_by_frame"][frame_idx], dtype=np.uint8), H, W).astype(bool)
    area_left = int(mask_left.sum())
    area_right = int(mask_right.sum())
    if area_left <= 0 or area_right <= 0:
        return None
    inter = int(np.logical_and(mask_left, mask_right).sum())
    union = int(np.logical_or(mask_left, mask_right).sum())
    mask_iou = 0.0 if union <= 0 else float(inter / float(union))
    containment = float(inter / float(min(area_left, area_right) + 1e-6))
    if mask_iou < float(args.trim_endpoint_min_mask_iou):
        return None
    if containment < float(args.trim_endpoint_min_containment):
        return None

    candidates: List[Tuple[float, int, int, float, int]] = []
    for idx, track in ((left_idx, left), (right_idx, right)):
        frames = _frames(track)
        endpoint_dist = _endpoint_distance(frames, frame_idx)
        if endpoint_dist > int(args.trim_endpoint_margin):
            continue
        area_ratio = float(track.get("area_by_frame", {}).get(frame_idx, 0.0))
        if area_ratio > float(args.trim_endpoint_max_area_ratio):
            continue
        peer = right if idx == left_idx else left
        if len(_frames(peer)) < int(args.trim_endpoint_min_peer_frames):
            continue
        # Prefer removing the smaller, closer-to-endpoint mask.
        candidates.append((area_ratio, endpoint_dist, int(idx), float(area_ratio), int(len(frames))))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row[0], row[1]))
    _score_area, endpoint_dist, remove_idx, remove_area_ratio, remove_frames = candidates[0]
    peer_idx = right_idx if remove_idx == left_idx else left_idx
    return {
        "remove_index": int(remove_idx),
        "peer_index": int(peer_idx),
        "frame_idx": int(frame_idx),
        "label": str(tracks[remove_idx].get("L_sem")),
        "mask_iou": float(mask_iou),
        "containment": float(containment),
        "remove_area_ratio": float(remove_area_ratio),
        "remove_endpoint_distance": int(endpoint_dist),
        "remove_frames": int(remove_frames),
        "peer_frames": int(len(_frames(tracks[peer_idx]))),
    }


def _trim_endpoint_duplicate_frames(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.trim_endpoint_duplicate_frames):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "trimmed_frames": 0,
            "dropped_empty_tracks": 0,
        }

    by_frame_label: Dict[Tuple[int, str], List[int]] = {}
    for idx, track in enumerate(tracks):
        if str(track.get("source_type")) != "thing_tracked":
            continue
        label = str(track.get("L_sem", "")).lower()
        for frame_idx in _frames(track):
            by_frame_label.setdefault((int(frame_idx), label), []).append(idx)

    drop_by_track: Dict[int, set[int]] = {}
    events: List[Dict[str, Any]] = []
    for (frame_idx, _label), indices in sorted(by_frame_label.items()):
        if len(indices) < 2:
            continue
        for pos, left_idx in enumerate(indices):
            for right_idx in indices[pos + 1 :]:
                match = _endpoint_duplicate_match(tracks, left_idx, right_idx, int(frame_idx), args, H, W)
                if match is None:
                    continue
                remove_idx = int(match["remove_index"])
                if int(frame_idx) in drop_by_track.get(remove_idx, set()):
                    continue
                drop_by_track.setdefault(remove_idx, set()).add(int(frame_idx))
                events.append(match)

    if not drop_by_track:
        return tracks, {
            "enabled": True,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "affected_tracks": 0,
            "trimmed_frames": 0,
            "dropped_empty_tracks": 0,
            "trim_endpoint_margin": int(args.trim_endpoint_margin),
            "trim_endpoint_min_mask_iou": float(args.trim_endpoint_min_mask_iou),
            "trim_endpoint_min_containment": float(args.trim_endpoint_min_containment),
            "trim_endpoint_max_area_ratio": float(args.trim_endpoint_max_area_ratio),
            "trim_endpoint_min_peer_frames": int(args.trim_endpoint_min_peer_frames),
            "label_counts_before": _counter_by_label(tracks),
            "label_counts_after": _counter_by_label(tracks),
            "short_thing_tracks_le3_before": _short_by_label(tracks),
            "short_thing_tracks_le3_after": _short_by_label(tracks),
            "trims_preview": [],
        }

    output: List[Dict[str, Any]] = []
    trims: List[Dict[str, Any]] = []
    dropped_empty = 0
    events_by_track: Dict[int, List[Dict[str, Any]]] = {}
    for event in events:
        events_by_track.setdefault(int(event["remove_index"]), []).append(event)

    for idx, track in enumerate(tracks):
        drop_frames = drop_by_track.get(idx, set())
        if not drop_frames:
            output.append(track)
            continue
        trimmed = _copy_without_frames(track, drop_frames)
        area_values = list(track.get("area_by_frame", {}).values())
        trims.append(
            {
                "track_index": int(idx),
                "label": str(track.get("L_sem")),
                "frames_before": _frames(track),
                "frames_trimmed": sorted(int(frame_idx) for frame_idx in drop_frames),
                "frames_after": [] if trimmed is None else _frames(trimmed),
                "max_area_ratio": float(max(area_values) if area_values else 0.0),
                "matches": events_by_track.get(idx, [])[:50],
            }
        )
        if trimmed is None:
            dropped_empty += 1
        else:
            output.append(trimmed)

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output)),
        "affected_tracks": int(len(trims)),
        "trimmed_frames": int(sum(len(row["frames_trimmed"]) for row in trims)),
        "dropped_empty_tracks": int(dropped_empty),
        "trim_endpoint_margin": int(args.trim_endpoint_margin),
        "trim_endpoint_min_mask_iou": float(args.trim_endpoint_min_mask_iou),
        "trim_endpoint_min_containment": float(args.trim_endpoint_min_containment),
        "trim_endpoint_max_area_ratio": float(args.trim_endpoint_max_area_ratio),
        "trim_endpoint_min_peer_frames": int(args.trim_endpoint_min_peer_frames),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output),
        "trims_preview": trims[:200],
    }
    return output, debug


def _drop_short_duplicate_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.drop_short_duplicate_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "dropped_tracks": 0,
        }
    kept: List[Dict[str, Any]] = []
    drops: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        if str(track.get("source_type")) != "thing_tracked" or len(_frames(track)) > int(args.drop_short_max_frames):
            kept.append(track)
            continue
        frame_matches: List[Dict[str, Any]] = []
        for frame_idx in _frames(track):
            match = _best_duplicate_peer(tracks, idx, int(frame_idx), args, H, W)
            if match is None:
                frame_matches = []
                break
            frame_matches.append(match)
        if frame_matches:
            area_values = list(track.get("area_by_frame", {}).values())
            drops.append(
                {
                    "track_index": int(idx),
                    "label": str(track.get("L_sem")),
                    "frames": _frames(track),
                    "max_area_ratio": float(max(area_values) if area_values else 0.0),
                    "matches": frame_matches,
                }
            )
        else:
            kept.append(track)
    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(kept)),
        "dropped_tracks": int(len(drops)),
        "drop_short_max_frames": int(args.drop_short_max_frames),
        "drop_duplicate_mask_iou": float(args.drop_duplicate_mask_iou),
        "drop_duplicate_box_iou": float(args.drop_duplicate_box_iou),
        "drop_duplicate_containment": float(args.drop_duplicate_containment),
        "drop_duplicate_containment_min_box_iou": float(args.drop_duplicate_containment_min_box_iou),
        "drop_duplicate_min_peer_frames": int(args.drop_duplicate_min_peer_frames),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(kept),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(kept),
        "drops_preview": drops[:200],
    }
    return kept, debug


def _best_short_fragment_peer(
    tracks: List[Dict[str, Any]],
    index: int,
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Optional[Dict[str, Any]]:
    track = tracks[index]
    label = str(track.get("L_sem", "")).lower()
    own_frames = _frames(track)
    min_peer_frames = max(1, int(np.ceil(len(own_frames) * float(args.drop_fragment_min_peer_ratio))))
    best: Optional[Dict[str, Any]] = None
    best_score: Tuple[int, float, float] = (-1, -1.0, -1.0)
    for peer_idx, peer in enumerate(tracks):
        if peer_idx == index:
            continue
        if str(peer.get("source_type")) != "thing_tracked":
            continue
        if str(peer.get("L_sem", "")).lower() != label:
            continue
        if len(_frames(peer)) < min_peer_frames:
            continue
        common = sorted(set(own_frames).intersection(_frames(peer)))
        if len(common) < int(args.drop_fragment_min_support_frames):
            continue
        support: List[Dict[str, Any]] = []
        for frame_idx in common:
            own_box = _as_box(track.get("box_by_frame", {}).get(frame_idx))
            peer_box = _as_box(peer.get("box_by_frame", {}).get(frame_idx))
            box_iou = _box_iou(own_box, peer_box)
            if box_iou < 0.05:
                continue
            mask_iou, containment, area_a, area_b, inter = _mask_iou_and_containment(track, peer, frame_idx, H, W)
            ok = mask_iou >= float(args.drop_fragment_min_mask_iou) or (
                containment >= float(args.drop_fragment_min_containment)
                and box_iou >= float(args.drop_fragment_min_box_iou)
            )
            if not ok:
                continue
            support.append(
                {
                    "frame_idx": int(frame_idx),
                    "mask_iou": float(mask_iou),
                    "containment": float(containment),
                    "box_iou": float(box_iou),
                    "own_area": int(area_a),
                    "peer_area": int(area_b),
                    "inter": int(inter),
                }
            )
        if len(support) < int(args.drop_fragment_min_support_frames):
            continue
        score = (
            int(len(support)),
            float(max(row["mask_iou"] for row in support)),
            float(max(row["containment"] for row in support)),
        )
        if score > best_score:
            best_score = score
            best = {
                "peer_index": int(peer_idx),
                "peer_frames": int(len(_frames(peer))),
                "support_frames": int(len(support)),
                "max_mask_iou": float(score[1]),
                "max_containment": float(score[2]),
                "support_preview": support[:50],
            }
    return best


def _drop_short_fragment_duplicate_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.drop_short_fragment_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "dropped_tracks": 0,
        }
    kept: List[Dict[str, Any]] = []
    drops: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        if str(track.get("source_type")) != "thing_tracked" or len(_frames(track)) > int(args.drop_fragment_max_frames):
            kept.append(track)
            continue
        match = _best_short_fragment_peer(tracks, idx, args, H, W)
        if match is None:
            kept.append(track)
            continue
        area_values = list(track.get("area_by_frame", {}).values())
        drops.append(
            {
                "track_index": int(idx),
                "label": str(track.get("L_sem")),
                "frames": _frames(track),
                "max_area_ratio": float(max(area_values) if area_values else 0.0),
                "match": match,
            }
        )
    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(kept)),
        "dropped_tracks": int(len(drops)),
        "drop_fragment_max_frames": int(args.drop_fragment_max_frames),
        "drop_fragment_min_peer_ratio": float(args.drop_fragment_min_peer_ratio),
        "drop_fragment_min_support_frames": int(args.drop_fragment_min_support_frames),
        "drop_fragment_min_mask_iou": float(args.drop_fragment_min_mask_iou),
        "drop_fragment_min_containment": float(args.drop_fragment_min_containment),
        "drop_fragment_min_box_iou": float(args.drop_fragment_min_box_iou),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(kept),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(kept),
        "drops_preview": drops[:200],
    }
    return kept, debug


def _edge_flicker_side(track: Dict[str, Any], W: int, margin_ratio: float) -> Optional[str]:
    frames = _frames(track)
    if not frames:
        return None
    margin_px = max(0.0, float(margin_ratio) * float(W))
    side: Optional[str] = None
    for frame_idx in frames:
        box = _as_box(track.get("box_by_frame", {}).get(int(frame_idx)))
        if box is None:
            return None
        x1, _y1, x2, _y2 = [float(v) for v in box]
        center_x = 0.5 * (x1 + x2)
        current: Optional[str]
        if center_x <= margin_px:
            current = "left"
        elif center_x >= float(W) - margin_px:
            current = "right"
        else:
            current = None
        if current is None:
            return None
        if side is None:
            side = current
        elif side != current:
            return None
    return side


def _drop_edge_flicker_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.drop_edge_flicker_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "dropped_tracks": 0,
        }
    labels = _label_set(args.drop_edge_flicker_labels)
    drops: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        frames = _frames(track)
        label = str(track.get("L_sem", "")).lower()
        if str(track.get("source_type")) != "thing_tracked" or (labels and label not in labels):
            continue
        if len(frames) > int(args.drop_edge_flicker_max_frames):
            continue
        if not frames or (max(frames) - min(frames) + 1) > int(args.drop_edge_flicker_max_span):
            continue
        area_values = [float(track.get("area_by_frame", {}).get(int(frame_idx), 0.0)) for frame_idx in frames]
        max_area = float(max(area_values) if area_values else 0.0)
        if max_area > float(args.drop_edge_flicker_max_area_ratio):
            continue
        side = _edge_flicker_side(track, W, float(args.drop_edge_flicker_margin_ratio))
        if side is None:
            continue
        drops.append(
            {
                "track_index": int(idx),
                "label": str(track.get("L_sem")),
                "frames": frames,
                "span": int(max(frames) - min(frames) + 1),
                "side": side,
                "max_area_ratio": max_area,
                "boxes_preview": [
                    [float(v) for v in _as_box(track.get("box_by_frame", {}).get(int(frame_idx)))]
                    for frame_idx in frames[:10]
                    if _as_box(track.get("box_by_frame", {}).get(int(frame_idx))) is not None
                ],
            }
        )
    dropped = {int(row["track_index"]) for row in drops}
    kept = []
    for idx, track in enumerate(tracks):
        if idx not in dropped:
            kept.append(track)
    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(kept)),
        "dropped_tracks": int(len(drops)),
        "drop_edge_flicker_labels": sorted(labels),
        "drop_edge_flicker_max_frames": int(args.drop_edge_flicker_max_frames),
        "drop_edge_flicker_max_span": int(args.drop_edge_flicker_max_span),
        "drop_edge_flicker_margin_ratio": float(args.drop_edge_flicker_margin_ratio),
        "drop_edge_flicker_max_area_ratio": float(args.drop_edge_flicker_max_area_ratio),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(kept),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(kept),
        "drops_preview": drops[:200],
    }
    return kept, debug


def _drop_short_label_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.drop_short_label_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "dropped_tracks": 0,
        }
    labels = _label_set(args.drop_short_labels)
    drops: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        frames = _frames(track)
        label = str(track.get("L_sem", "")).lower()
        if str(track.get("source_type")) != "thing_tracked" or (labels and label not in labels):
            continue
        if len(frames) > int(args.drop_short_label_max_frames):
            continue
        if not frames or (max(frames) - min(frames) + 1) > int(args.drop_short_label_max_span):
            continue
        area_values = [float(track.get("area_by_frame", {}).get(int(frame_idx), 0.0)) for frame_idx in frames]
        max_area = float(max(area_values) if area_values else 0.0)
        if max_area > float(args.drop_short_label_max_area_ratio):
            continue
        drops.append(
            {
                "track_index": int(idx),
                "label": str(track.get("L_sem")),
                "frames": frames,
                "span": int(max(frames) - min(frames) + 1),
                "max_area_ratio": max_area,
                "boxes_preview": [
                    [float(v) for v in _as_box(track.get("box_by_frame", {}).get(int(frame_idx)))]
                    for frame_idx in frames[:10]
                    if _as_box(track.get("box_by_frame", {}).get(int(frame_idx))) is not None
                ],
            }
        )
    dropped = {int(row["track_index"]) for row in drops}
    kept = [track for idx, track in enumerate(tracks) if idx not in dropped]
    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(kept)),
        "dropped_tracks": int(len(drops)),
        "drop_short_labels": sorted(labels),
        "drop_short_label_max_frames": int(args.drop_short_label_max_frames),
        "drop_short_label_max_span": int(args.drop_short_label_max_span),
        "drop_short_label_max_area_ratio": float(args.drop_short_label_max_area_ratio),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(kept),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(kept),
        "drops_preview": drops[:200],
    }
    return kept, debug


def _interpolate_anchor_box(
    anchor: Dict[str, Any],
    frame_idx: int,
    max_neighbor_gap: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    frames = _frames(anchor)
    if not frames:
        return None, {}
    frame_idx = int(frame_idx)
    before = [f for f in frames if int(f) < frame_idx]
    after = [f for f in frames if int(f) > frame_idx]
    left_f = max(before) if before else None
    right_f = min(after) if after else None
    left_gap = None if left_f is None else int(frame_idx - left_f)
    right_gap = None if right_f is None else int(right_f - frame_idx)
    if (left_gap is None or left_gap > int(max_neighbor_gap)) and (
        right_gap is None or right_gap > int(max_neighbor_gap)
    ):
        return None, {"left_frame": left_f, "right_frame": right_f, "left_gap": left_gap, "right_gap": right_gap}

    left_box = _as_box(anchor.get("box_by_frame", {}).get(int(left_f))) if left_f is not None else None
    right_box = _as_box(anchor.get("box_by_frame", {}).get(int(right_f))) if right_f is not None else None
    if left_box is not None and right_box is not None:
        denom = max(int(right_f - left_f), 1)
        alpha = float(frame_idx - left_f) / float(denom)
        pred = left_box * (1.0 - alpha) + right_box * alpha
        mode = "interp"
    elif left_box is not None:
        pred = left_box
        mode = "left"
    elif right_box is not None:
        pred = right_box
        mode = "right"
    else:
        return None, {"left_frame": left_f, "right_frame": right_f, "left_gap": left_gap, "right_gap": right_gap}
    return pred.astype(np.float32), {
        "left_frame": left_f,
        "right_frame": right_f,
        "left_gap": left_gap,
        "right_gap": right_gap,
        "mode": mode,
    }


def _gap_fill_short_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.gap_fill_short_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }

    labels = _label_set(args.gap_fill_short_labels)
    max_short_frames = int(args.gap_fill_short_max_frames)
    anchor_min_frames = int(args.gap_fill_anchor_min_frames)
    max_neighbor_gap = int(args.gap_fill_max_neighbor_gap)
    max_center_dist = float(args.gap_fill_max_center_dist)
    min_mean_iou = float(args.gap_fill_min_mean_iou)
    max_area_ratio = float(args.gap_fill_short_max_area_ratio)
    center_floor = float(args.gap_fill_center_scale_floor)

    n = len(tracks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(anchor_idx: int, short_idx: int) -> bool:
        ra, rb = find(anchor_idx), find(short_idx)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    track_frames = [_frames(track) for track in tracks]
    track_frame_sets = [set(frames) for frames in track_frames]
    short_indices: List[int] = []
    anchor_indices: List[int] = []
    for idx, track in enumerate(tracks):
        if str(track.get("source_type")) != "thing_tracked":
            continue
        label = str(track.get("L_sem", "")).lower()
        if labels and label not in labels:
            continue
        frames = track_frames[idx]
        if not frames:
            continue
        if len(frames) >= anchor_min_frames:
            anchor_indices.append(idx)
        if len(frames) <= max_short_frames:
            areas = [float(track.get("area_by_frame", {}).get(int(frame_idx), 0.0)) for frame_idx in frames]
            if float(max(areas) if areas else 0.0) <= max_area_ratio:
                short_indices.append(idx)

    decisions: List[Dict[str, Any]] = []
    rejected_preview: List[Dict[str, Any]] = []
    used_shorts: set[int] = set()
    for short_idx in short_indices:
        short = tracks[short_idx]
        short_label = str(short.get("L_sem", "")).lower()
        short_frames = track_frames[short_idx]
        short_start, short_end = int(short_frames[0]), int(short_frames[-1])
        best: Optional[Dict[str, Any]] = None
        for anchor_idx in anchor_indices:
            if anchor_idx == short_idx:
                continue
            anchor = tracks[anchor_idx]
            if str(anchor.get("L_sem", "")).lower() != short_label:
                continue
            anchor_frames = track_frames[anchor_idx]
            if not anchor_frames:
                continue
            if track_frame_sets[anchor_idx].intersection(track_frame_sets[short_idx]):
                continue
            if short_start < int(anchor_frames[0]) - max_neighbor_gap or short_end > int(anchor_frames[-1]) + max_neighbor_gap:
                continue

            per_frame: List[Dict[str, Any]] = []
            dists: List[float] = []
            ious: List[float] = []
            for frame_idx in short_frames:
                short_box = _as_box(short.get("box_by_frame", {}).get(int(frame_idx)))
                pred_box, pred_info = _interpolate_anchor_box(anchor, int(frame_idx), max_neighbor_gap)
                if short_box is None or pred_box is None:
                    continue
                dist = _center_dist_norm_floor(pred_box, short_box, center_floor)
                box_iou = _box_iou(pred_box, short_box)
                dists.append(float(dist))
                ious.append(float(box_iou))
                per_frame.append(
                    {
                        "frame": int(frame_idx),
                        "center_dist": float(dist),
                        "box_iou": float(box_iou),
                        **pred_info,
                    }
                )
            if not dists:
                continue
            mean_dist = float(np.mean(dists))
            max_dist = float(np.max(dists))
            mean_iou = float(np.mean(ious)) if ious else 0.0
            ok = bool(mean_dist <= max_center_dist and max_dist <= max_center_dist * 1.5 and mean_iou >= min_mean_iou)
            row = {
                "anchor_index": int(anchor_idx),
                "short_index": int(short_idx),
                "label": short_label,
                "short_frames": [int(f) for f in short_frames],
                "anchor_range": [int(anchor_frames[0]), int(anchor_frames[-1])],
                "anchor_frames": int(len(anchor_frames)),
                "short_frames_count": int(len(short_frames)),
                "mean_center_dist": mean_dist,
                "max_center_dist": max_dist,
                "mean_box_iou": mean_iou,
                "per_frame": per_frame[:10],
            }
            row["score"] = float(mean_dist - 0.25 * mean_iou)
            if ok:
                if best is None or float(row["score"]) < float(best["score"]):
                    best = row
            elif len(rejected_preview) < 100:
                rejected_preview.append(row)
        if best is None or short_idx in used_shorts:
            continue
        if union(int(best["anchor_index"]), short_idx):
            used_shorts.add(short_idx)
            decisions.append(best)

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        output_tracks.append(_merge_members([tracks[idx] for idx in indices]))

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "merged_tracks": int(len(tracks) - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "merge_pairs": int(len(decisions)),
        "gap_fill_short_labels": sorted(labels),
        "gap_fill_short_max_frames": int(max_short_frames),
        "gap_fill_anchor_min_frames": int(anchor_min_frames),
        "gap_fill_max_neighbor_gap": int(max_neighbor_gap),
        "gap_fill_max_center_dist": float(max_center_dist),
        "gap_fill_min_mean_iou": float(min_mean_iou),
        "gap_fill_short_max_area_ratio": float(max_area_ratio),
        "gap_fill_center_scale_floor": float(center_floor),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output_tracks),
        "decisions_preview": decisions[:200],
        "rejected_preview": rejected_preview,
    }
    return output_tracks, debug


def _adjacent_flicker_common_overlap_ok(
    left: Dict[str, Any],
    right: Dict[str, Any],
    common_frames: List[int],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[bool, Dict[str, Any]]:
    if not common_frames:
        return True, {
            "common_frames": 0,
            "max_mask_iou": 0.0,
            "max_containment": 0.0,
            "mean_mask_iou": 0.0,
            "mean_containment": 0.0,
        }

    mask_ious: List[float] = []
    containments: List[float] = []
    for frame_idx in common_frames:
        mask_iou, containment, _area_left, _area_right, _inter = _mask_iou_and_containment(
            left,
            right,
            int(frame_idx),
            H,
            W,
        )
        mask_ious.append(float(mask_iou))
        containments.append(float(containment))

    max_mask_iou = float(max(mask_ious) if mask_ious else 0.0)
    max_containment = float(max(containments) if containments else 0.0)
    mean_mask_iou = float(np.mean(mask_ious)) if mask_ious else 0.0
    mean_containment = float(np.mean(containments)) if containments else 0.0
    max_common = int(args.adjacent_flicker_max_common_frames)
    ok = bool(
        len(common_frames) <= max_common
        and (
            max_mask_iou >= float(args.adjacent_flicker_common_min_mask_iou)
            or max_containment >= float(args.adjacent_flicker_common_min_containment)
        )
    )
    return ok, {
        "common_frames": int(len(common_frames)),
        "common_frame_preview": [int(f) for f in common_frames[:20]],
        "max_mask_iou": max_mask_iou,
        "max_containment": max_containment,
        "mean_mask_iou": mean_mask_iou,
        "mean_containment": mean_containment,
    }


def _adjacent_flicker_merge_tracks(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.adjacent_flicker_merge_tracks):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }

    labels = _label_set(args.adjacent_flicker_labels)
    n = len(tracks)
    track_frames = [_frames(track) for track in tracks]
    track_frame_sets = [set(frames) for frames in track_frames]
    active_by_frame_label: Dict[Tuple[int, str], List[int]] = {}
    max_frame = -1
    for idx, track in enumerate(tracks):
        if str(track.get("source_type")) != "thing_tracked":
            continue
        label = str(track.get("L_sem", "")).lower()
        if labels and label not in labels:
            continue
        for frame_idx in track_frames[idx]:
            active_by_frame_label.setdefault((int(frame_idx), label), []).append(idx)
            max_frame = max(max_frame, int(frame_idx))

    supports: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    event_rows_by_frame_label: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    rejected_preview: List[Dict[str, Any]] = []
    for frame_idx in range(max_frame):
        for label in labels or {str(track.get("L_sem", "")).lower() for track in tracks}:
            left_active = active_by_frame_label.get((frame_idx, label), [])
            right_active = active_by_frame_label.get((frame_idx + 1, label), [])
            if not left_active or not right_active:
                continue
            for left_idx in left_active:
                for right_idx in right_active:
                    if left_idx == right_idx:
                        continue
                    if right_idx in active_by_frame_label.get((frame_idx, label), []):
                        continue
                    if left_idx in active_by_frame_label.get((frame_idx + 1, label), []):
                        continue
                    left = tracks[left_idx]
                    right = tracks[right_idx]
                    left_box = _as_box(left.get("box_by_frame", {}).get(int(frame_idx)))
                    right_box = _as_box(right.get("box_by_frame", {}).get(int(frame_idx + 1)))
                    if left_box is None or right_box is None:
                        continue
                    box_iou = _box_iou(left_box, right_box)
                    center = _center_dist_norm_floor(
                        left_box,
                        right_box,
                        float(args.adjacent_flicker_center_scale_floor),
                    )
                    area_ratio = _area_ratio_pair(left, int(frame_idx), right, int(frame_idx + 1))
                    row = {
                        "frame_left": int(frame_idx),
                        "frame_right": int(frame_idx + 1),
                        "left_index": int(left_idx),
                        "right_index": int(right_idx),
                        "label": label,
                        "box_iou": float(box_iou),
                        "center_dist": float(center),
                        "area_ratio": float(area_ratio),
                        "left_frames": int(len(track_frames[left_idx])),
                        "right_frames": int(len(track_frames[right_idx])),
                        "left_range": [
                            int(track_frames[left_idx][0]) if track_frames[left_idx] else -1,
                            int(track_frames[left_idx][-1]) if track_frames[left_idx] else -1,
                        ],
                        "right_range": [
                            int(track_frames[right_idx][0]) if track_frames[right_idx] else -1,
                            int(track_frames[right_idx][-1]) if track_frames[right_idx] else -1,
                        ],
                    }
                    row["event_score"] = float(
                        0.50 * (1.0 / (1.0 + float(center)))
                        + 0.25 * min(1.0, float(box_iou) / 0.20)
                        + 0.25 * min(1.0, float(area_ratio))
                    )
                    regular_candidate = bool(
                        area_ratio >= float(args.adjacent_flicker_min_area_ratio)
                        and (
                            box_iou >= float(args.adjacent_flicker_min_box_iou)
                            or center <= float(args.adjacent_flicker_max_center_dist)
                        )
                    )
                    single_strong_candidate = bool(
                        bool(int(args.adjacent_flicker_allow_single_strong))
                        and area_ratio >= float(args.adjacent_flicker_single_min_area_ratio)
                        and (
                            box_iou >= float(args.adjacent_flicker_single_min_box_iou)
                            or center <= float(args.adjacent_flicker_single_max_center_dist)
                        )
                    )
                    row["candidate_gate"] = (
                        "regular"
                        if regular_candidate
                        else "single_strong"
                        if single_strong_candidate
                        else "rejected"
                    )
                    if regular_candidate or single_strong_candidate:
                        event_rows_by_frame_label.setdefault((int(frame_idx), label), []).append(row)
                    elif len(rejected_preview) < 100:
                        rejected_preview.append(row)

    for (_frame_label, rows) in event_rows_by_frame_label.items():
        best_by_left: Dict[int, Dict[str, Any]] = {}
        best_by_right: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            left_idx = int(row["left_index"])
            right_idx = int(row["right_index"])
            if (
                left_idx not in best_by_left
                or float(row["event_score"]) > float(best_by_left[left_idx]["event_score"])
            ):
                best_by_left[left_idx] = row
            if (
                right_idx not in best_by_right
                or float(row["event_score"]) > float(best_by_right[right_idx]["event_score"])
            ):
                best_by_right[right_idx] = row
        for row in rows:
            left_idx = int(row["left_index"])
            right_idx = int(row["right_index"])
            row["mutual_nearest"] = bool(
                best_by_left.get(left_idx) is row and best_by_right.get(right_idx) is row
            )
            key = (min(left_idx, right_idx), max(left_idx, right_idx))
            supports.setdefault(key, []).append(row)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        return True

    decisions: List[Dict[str, Any]] = []
    accepted_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for (left_idx, right_idx), events in sorted(
        supports.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    ):
        single_event = events[0] if len(events) == 1 else None
        single_strong = bool(
            single_event is not None
            and bool(int(args.adjacent_flicker_allow_single_strong))
            and bool(single_event.get("mutual_nearest", False))
            and float(single_event.get("area_ratio", 0.0)) >= float(args.adjacent_flicker_single_min_area_ratio)
            and (
                float(single_event.get("box_iou", 0.0)) >= float(args.adjacent_flicker_single_min_box_iou)
                or float(single_event.get("center_dist", 999.0)) <= float(args.adjacent_flicker_single_max_center_dist)
            )
        )
        has_min_support = len(events) >= int(args.adjacent_flicker_min_support)
        if not has_min_support and not single_strong:
            if len(rejected_preview) < 200:
                rejected_preview.append({
                    "left_index": int(left_idx),
                    "right_index": int(right_idx),
                    "label": str(tracks[left_idx].get("L_sem")),
                    "support_count": int(len(events)),
                    "reason": "insufficient_support_not_single_strong",
                    "events_preview": events[:20],
                })
            continue
        if find(left_idx) == find(right_idx):
            continue
        common = sorted(track_frame_sets[left_idx].intersection(track_frame_sets[right_idx]))
        common_ok, common_info = _adjacent_flicker_common_overlap_ok(
            tracks[left_idx],
            tracks[right_idx],
            common,
            args,
            H,
            W,
        )
        decision = {
            "left_index": int(left_idx),
            "right_index": int(right_idx),
            "label": str(tracks[left_idx].get("L_sem")),
            "support_count": int(len(events)),
            "single_strong": bool(single_strong),
            "has_min_support": bool(has_min_support),
            "events_preview": events[:20],
            **common_info,
        }
        if not common_ok:
            decision["reason"] = "common_frame_overlap_guard"
            rejected_preview.append(decision)
            continue
        union(left_idx, right_idx)
        decision["reason"] = "adjacent_flicker_single_strong" if single_strong and not has_min_support else "adjacent_flicker"
        decisions.append(decision)
        accepted_by_pair[(min(left_idx, right_idx), max(left_idx, right_idx))] = decision

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        group_decisions = [
            decision
            for (a, b), decision in accepted_by_pair.items()
            if a in indices and b in indices
        ]
        history_event = {
            "op": "adjacent_flicker_identity_stitch",
            "decision_count": int(len(group_decisions)),
            "decisions": group_decisions[:20],
            "member_indices": [int(idx) for idx in indices],
        }
        output_tracks.append(_merge_members([tracks[idx] for idx in indices], history_event=history_event))

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "merged_tracks": int(len(tracks) - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "candidate_pairs": int(len(supports)),
        "merge_pairs": int(len(decisions)),
        "adjacent_flicker_labels": sorted(labels),
        "adjacent_flicker_min_support": int(args.adjacent_flicker_min_support),
        "adjacent_flicker_min_box_iou": float(args.adjacent_flicker_min_box_iou),
        "adjacent_flicker_max_center_dist": float(args.adjacent_flicker_max_center_dist),
        "adjacent_flicker_min_area_ratio": float(args.adjacent_flicker_min_area_ratio),
        "adjacent_flicker_center_scale_floor": float(args.adjacent_flicker_center_scale_floor),
        "adjacent_flicker_max_common_frames": int(args.adjacent_flicker_max_common_frames),
        "adjacent_flicker_common_min_mask_iou": float(args.adjacent_flicker_common_min_mask_iou),
        "adjacent_flicker_common_min_containment": float(args.adjacent_flicker_common_min_containment),
        "adjacent_flicker_allow_single_strong": bool(int(args.adjacent_flicker_allow_single_strong)),
        "adjacent_flicker_single_min_box_iou": float(args.adjacent_flicker_single_min_box_iou),
        "adjacent_flicker_single_max_center_dist": float(args.adjacent_flicker_single_max_center_dist),
        "adjacent_flicker_single_min_area_ratio": float(args.adjacent_flicker_single_min_area_ratio),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output_tracks),
        "decisions_preview": decisions[:200],
        "rejected_preview": rejected_preview[:200],
    }
    return output_tracks, debug


def _parse_manual_merge_track_groups(raw: str) -> List[List[int]]:
    groups: List[List[int]] = []
    for raw_group in str(raw or "").replace("|", ";").split(";"):
        group: List[int] = []
        for item in raw_group.replace("+", ",").split(","):
            item = item.strip()
            if not item:
                continue
            group.append(int(item))
        group = sorted(dict.fromkeys(group))
        if len(group) >= 2:
            groups.append(group)
    return groups


def _manual_merge_track_groups(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    requested_groups = _parse_manual_merge_track_groups(str(getattr(args, "manual_merge_track_groups", "") or ""))
    if not requested_groups:
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "merged_tracks": 0,
        }

    n = len(tracks)
    parent = list(range(n))
    require_no_common = bool(int(getattr(args, "manual_merge_require_no_common_frames", 1)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        return True

    accepted: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for group in requested_groups:
        item: Dict[str, Any] = {"indices": [int(idx) for idx in group]}
        invalid = [int(idx) for idx in group if idx < 0 or idx >= n]
        if invalid:
            item["reason"] = "invalid_index"
            item["invalid_indices"] = invalid
            skipped.append(item)
            continue

        labels = {str(tracks[idx].get("L_sem", "")) for idx in group}
        source_types = {str(tracks[idx].get("source_type", "")) for idx in group}
        if len(labels) != 1:
            item["reason"] = "label_mismatch"
            item["labels"] = sorted(labels)
            skipped.append(item)
            continue
        if len(source_types) != 1:
            item["reason"] = "source_type_mismatch"
            item["source_types"] = sorted(source_types)
            skipped.append(item)
            continue

        frame_sets = [set(_frames(tracks[idx])) for idx in group]
        common_frames = sorted(set.intersection(*frame_sets)) if frame_sets else []
        if require_no_common and common_frames:
            item["reason"] = "common_frames_present"
            item["common_frames_preview"] = [int(frame) for frame in common_frames[:20]]
            skipped.append(item)
            continue

        for idx in group[1:]:
            union(group[0], idx)
        frame_ranges = []
        for idx in group:
            frames = _frames(tracks[idx])
            frame_ranges.append(
                {
                    "index": int(idx),
                    "frames": int(len(frames)),
                    "start": int(frames[0]) if frames else None,
                    "end": int(frames[-1]) if frames else None,
                }
            )
        item.update(
            {
                "label": next(iter(labels)),
                "source_type": next(iter(source_types)),
                "frame_ranges": frame_ranges,
                "common_frames_count": int(len(common_frames)),
            }
        )
        accepted.append(item)

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        output_tracks.append(_merge_members([tracks[idx] for idx in indices]))

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "merged_tracks": int(len(tracks) - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "requested_groups": requested_groups,
        "accepted_groups": accepted,
        "skipped_groups": skipped,
        "require_no_common_frames": require_no_common,
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output_tracks),
    }
    return output_tracks, debug


def _track_mask_coverage(track: Dict[str, Any], frame_idx: int, proposal_mask: np.ndarray, H: int, W: int) -> float:
    packed = track.get("mask_by_frame", {}).get(int(frame_idx))
    if packed is None:
        return 0.0
    mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
    area = int(proposal_mask.sum())
    if area <= 0:
        return 0.0
    return float(np.logical_and(mask, proposal_mask).sum() / float(area))


def _union_coverage_for_label(
    tracks: List[Dict[str, Any]],
    frame_idx: int,
    proposal_mask: np.ndarray,
    label: str,
    H: int,
    W: int,
) -> float:
    union = np.zeros((H, W), dtype=bool)
    for track in tracks:
        if str(track.get("L_sem", "")).lower() != label.lower():
            continue
        if "thing" not in str(track.get("source_type", "")).lower():
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        union |= _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W).astype(bool)
    area = int(proposal_mask.sum())
    if area <= 0:
        return 0.0
    return float(np.logical_and(union, proposal_mask).sum() / float(area))


def _postmerge_proposal_frame_repair(
    tracks: List[Dict[str, Any]],
    args: argparse.Namespace,
    H: int,
    W: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not int(args.postmerge_proposal_repair):
        return tracks, {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "repaired_frames": 0,
        }
    required = {
        "proposal_tracklets_pt": str(args.proposal_tracklets_pt or ""),
        "proposal_chunks_root": str(args.proposal_chunks_root or ""),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        return tracks, {
            "enabled": True,
            "skipped": f"missing_args:{','.join(missing)}",
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "repaired_frames": 0,
        }

    proposal_tracklets_path = Path(args.proposal_tracklets_pt)
    proposal_chunks_root = Path(args.proposal_chunks_root)
    if not proposal_tracklets_path.exists() or not proposal_chunks_root.exists():
        return tracks, {
            "enabled": True,
            "skipped": "missing_files",
            "proposal_tracklets_pt": str(proposal_tracklets_path),
            "proposal_chunks_root": str(proposal_chunks_root),
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
            "repaired_frames": 0,
        }

    labels = _label_set(args.postmerge_repair_labels)
    target_tracklet_ids = {int(item) for item in _label_set(args.postmerge_repair_tracklet_ids)}
    tracklet_payload = torch.load(proposal_tracklets_path, map_location="cpu", weights_only=False)
    tracklet_to_proposal_ids = {
        int(rec["tracklet_id"]): [int(pid) for pid in rec.get("proposal_ids", [])]
        for rec in tracklet_payload.get("tracklets", [])
    }
    proposals_by_id = _load_proposals_by_id(proposal_chunks_root)
    output = list(tracks)
    repairs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for tracklet_id in sorted(target_tracklet_ids):
        proposal_ids = tracklet_to_proposal_ids.get(int(tracklet_id), [])
        proposals: List[Dict[str, Any]] = []
        for proposal_id in proposal_ids:
            proposal = proposals_by_id.get(int(proposal_id))
            if proposal is None:
                continue
            label = str(proposal.get("label", "")).lower()
            if labels and label not in labels:
                continue
            proposal_mask = _proposal_mask(proposal, H, W)
            proposal_box = _proposal_box(proposal)
            if proposal_mask is None or proposal_box is None:
                continue
            area_ratio = _proposal_area_ratio(proposal, proposal_mask)
            if area_ratio < float(args.postmerge_repair_min_area_ratio) or area_ratio > float(args.postmerge_repair_max_area_ratio):
                continue
            item = dict(proposal)
            item["_proposal_mask"] = proposal_mask
            item["_proposal_box"] = proposal_box
            item["_area_ratio"] = float(area_ratio)
            proposals.append(item)
        if not proposals:
            skipped.append({"proposal_tracklet_id": int(tracklet_id), "reason": "no_matching_proposals"})
            continue

        support_by_track: Dict[int, Dict[str, Any]] = {}
        for proposal in proposals:
            conf = float(proposal.get("confidence", 0.0))
            if conf < float(args.postmerge_repair_support_min_conf):
                continue
            frame_idx = int(proposal.get("frame_idx", -1))
            label = str(proposal.get("label", "")).lower()
            proposal_mask = proposal["_proposal_mask"]
            proposal_box = proposal["_proposal_box"]
            for track_idx, track in enumerate(output):
                if str(track.get("L_sem", "")).lower() != label:
                    continue
                if "thing" not in str(track.get("source_type", "")).lower():
                    continue
                if int(frame_idx) not in track.get("mask_by_frame", {}):
                    continue
                coverage = _track_mask_coverage(track, frame_idx, proposal_mask, H, W)
                box_iou = _box_iou(_as_box(track.get("box_by_frame", {}).get(frame_idx)), proposal_box)
                if coverage < float(args.postmerge_repair_min_support_coverage) and box_iou < float(args.postmerge_repair_min_support_box_iou):
                    continue
                state = support_by_track.setdefault(
                    track_idx,
                    {
                        "support_frames": set(),
                        "score": 0.0,
                        "preview": [],
                    },
                )
                state["support_frames"].add(int(frame_idx))
                state["score"] += max(float(coverage), float(box_iou))
                if len(state["preview"]) < 20:
                    state["preview"].append(
                        {
                            "frame_idx": int(frame_idx),
                            "proposal_id": int(proposal.get("proposal_id", -1)),
                            "confidence": float(conf),
                            "coverage": float(coverage),
                            "box_iou": float(box_iou),
                        }
                    )

        if not support_by_track:
            skipped.append({"proposal_tracklet_id": int(tracklet_id), "reason": "no_supported_output_track"})
            continue
        best_track_idx, best_state = max(
            support_by_track.items(),
            key=lambda item: (len(item[1]["support_frames"]), float(item[1]["score"])),
        )
        if len(best_state["support_frames"]) < int(args.postmerge_repair_min_support_frames):
            skipped.append(
                {
                    "proposal_tracklet_id": int(tracklet_id),
                    "reason": "insufficient_support",
                    "best_track_index": int(best_track_idx),
                    "support_frames": int(len(best_state["support_frames"])),
                    "support_preview": best_state["preview"],
                }
            )
            continue

        target = output[best_track_idx]
        for proposal in proposals:
            conf = float(proposal.get("confidence", 0.0))
            if conf < float(args.postmerge_repair_min_conf):
                continue
            frame_idx = int(proposal.get("frame_idx", -1))
            label = str(proposal.get("label", "")).lower()
            proposal_mask = proposal["_proposal_mask"]
            proposal_box = proposal["_proposal_box"]
            union_coverage = _union_coverage_for_label(output, frame_idx, proposal_mask, label, H, W)
            if union_coverage > float(args.postmerge_repair_max_coverage):
                continue
            target_coverage = _track_mask_coverage(target, frame_idx, proposal_mask, H, W)
            had_frame = int(frame_idx) in target.get("mask_by_frame", {})
            target["mask_by_frame"][frame_idx] = _pack_mask_np(proposal_mask)
            target["box_by_frame"][frame_idx] = torch.tensor(proposal_box, dtype=torch.float32)
            target["q_by_frame"][frame_idx] = float(conf)
            target["area_by_frame"][frame_idx] = float(proposal["_area_ratio"])
            if int(frame_idx) < int(target.get("birth_frame", frame_idx)):
                target["birth_frame"] = int(frame_idx)
            repairs.append(
                {
                    "proposal_tracklet_id": int(tracklet_id),
                    "target_track_index": int(best_track_idx),
                    "proposal_id": int(proposal.get("proposal_id", -1)),
                    "frame_idx": int(frame_idx),
                    "label": label,
                    "confidence": float(conf),
                    "area_ratio": float(proposal["_area_ratio"]),
                    "union_coverage_before": float(union_coverage),
                    "target_coverage_before": float(target_coverage),
                    "action": "replace_frame" if had_frame else "add_frame",
                    "support_frames": int(len(best_state["support_frames"])),
                }
            )

    debug = {
        "enabled": True,
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output)),
        "repaired_frames": int(len(repairs)),
        "repaired_tracks": int(len({row["target_track_index"] for row in repairs})),
        "proposal_tracklets_pt": str(proposal_tracklets_path),
        "proposal_chunks_root": str(proposal_chunks_root),
        "postmerge_repair_labels": sorted(labels),
        "postmerge_repair_tracklet_ids": sorted(target_tracklet_ids),
        "postmerge_repair_min_conf": float(args.postmerge_repair_min_conf),
        "postmerge_repair_support_min_conf": float(args.postmerge_repair_support_min_conf),
        "postmerge_repair_max_coverage": float(args.postmerge_repair_max_coverage),
        "postmerge_repair_min_area_ratio": float(args.postmerge_repair_min_area_ratio),
        "postmerge_repair_max_area_ratio": float(args.postmerge_repair_max_area_ratio),
        "postmerge_repair_min_support_frames": int(args.postmerge_repair_min_support_frames),
        "postmerge_repair_min_support_coverage": float(args.postmerge_repair_min_support_coverage),
        "postmerge_repair_min_support_box_iou": float(args.postmerge_repair_min_support_box_iou),
        "repairs_preview": repairs[:200],
        "skipped_preview": skipped[:100],
    }
    return output, debug


def merge_sparse(sparse: Any, args: argparse.Namespace, image_paths: Optional[List[str]] = None) -> tuple[Any, Dict[str, Any]]:
    merged = clone_sparse(sparse)
    tracks = merged.tracks
    H, W = int(merged.frame_height), int(merged.frame_width)
    tracks, canonicalize_vehicle_debug = _canonicalize_vehicle_labels(tracks, args)
    tracks, proposal_drift_repair_debug = _repair_sam3_drift_with_proposals(tracks, args, H, W)
    tracks, proposal_miss_fallback_debug = _add_missed_proposal_fallback_tracks(tracks, args, H, W)
    n = len(tracks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        parent[rb] = ra
        return True

    decisions: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(i + 1, n):
            ok, reason, score = _pair_decision(tracks[i], tracks[j], H, W, args)
            if ok and union(i, j):
                decisions.append(
                    {
                        "from_index": int(j),
                        "to_index": int(i),
                        "label": str(tracks[i].get("L_sem")),
                        "reason": reason,
                        "score": float(score),
                    }
                )

    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        groups.setdefault(find(idx), []).append(idx)

    output_tracks: List[Dict[str, Any]] = []
    merged_groups = 0
    for root in sorted(groups):
        indices = groups[root]
        if len(indices) == 1:
            output_tracks.append(tracks[indices[0]])
            continue
        merged_groups += 1
        output_tracks.append(_merge_members([tracks[idx] for idx in indices]))

    output_tracks_after_merge = len(output_tracks)
    output_tracks, reid_debug = _reid_merge_tracks(output_tracks, args, H, W, image_paths)
    output_tracks_after_reid = len(output_tracks)
    output_tracks, overlap_support_debug = _overlap_support_merge_tracks(output_tracks, args, H, W)
    output_tracks_after_overlap_support = len(output_tracks)
    output_tracks, trim_debug = _trim_duplicate_frames(output_tracks, args, H, W)
    output_tracks_after_trim = len(output_tracks)
    output_tracks, contain_trim_debug = _trim_contained_duplicate_frames(output_tracks, args, H, W)
    output_tracks_after_contain_trim = len(output_tracks)
    output_tracks, endpoint_trim_debug = _trim_endpoint_duplicate_frames(output_tracks, args, H, W)
    output_tracks_after_endpoint_trim = len(output_tracks)
    output_tracks, fragment_drop_debug = _drop_short_fragment_duplicate_tracks(output_tracks, args, H, W)
    output_tracks_after_fragment_drop = len(output_tracks)
    output_tracks, drop_debug = _drop_short_duplicate_tracks(output_tracks, args, H, W)
    output_tracks, postmerge_proposal_repair_debug = _postmerge_proposal_frame_repair(output_tracks, args, H, W)
    output_tracks_after_postmerge_repair = len(output_tracks)
    output_tracks, temporal_link_debug = _temporal_link_tracks_multi(output_tracks, args)
    output_tracks_after_temporal_link = len(output_tracks)
    output_tracks, edge_flicker_drop_debug = _drop_edge_flicker_tracks(output_tracks, args, H, W)
    output_tracks_after_edge_flicker = len(output_tracks)
    output_tracks, short_label_drop_debug = _drop_short_label_tracks(output_tracks, args, H, W)
    output_tracks_after_short_label_drop = len(output_tracks)
    output_tracks, gap_fill_debug = _gap_fill_short_tracks(output_tracks, args)
    output_tracks_after_gap_fill = len(output_tracks)
    output_tracks, adjacent_flicker_debug = _adjacent_flicker_merge_tracks(output_tracks, args, H, W)
    output_tracks_after_adjacent_flicker = len(output_tracks)
    output_tracks, manual_merge_debug = _manual_merge_track_groups(output_tracks, args)

    debug = {
        "input_tracks": int(len(tracks)),
        "output_tracks": int(len(output_tracks)),
        "output_tracks_after_merge": int(output_tracks_after_merge),
        "output_tracks_after_reid": int(output_tracks_after_reid),
        "output_tracks_after_overlap_support": int(output_tracks_after_overlap_support),
        "output_tracks_after_trim": int(output_tracks_after_trim),
        "output_tracks_after_contain_trim": int(output_tracks_after_contain_trim),
        "output_tracks_after_endpoint_trim": int(output_tracks_after_endpoint_trim),
        "output_tracks_after_fragment_drop": int(output_tracks_after_fragment_drop),
        "output_tracks_after_postmerge_repair": int(output_tracks_after_postmerge_repair),
        "output_tracks_after_temporal_link": int(output_tracks_after_temporal_link),
        "output_tracks_after_edge_flicker": int(output_tracks_after_edge_flicker),
        "output_tracks_after_short_label_drop": int(output_tracks_after_short_label_drop),
        "output_tracks_after_gap_fill": int(output_tracks_after_gap_fill),
        "output_tracks_after_adjacent_flicker": int(output_tracks_after_adjacent_flicker),
        "merged_tracks": int(len(tracks) - output_tracks_after_merge),
        "reid_merged_tracks": int(output_tracks_after_merge - output_tracks_after_reid),
        "overlap_support_merged_tracks": int(output_tracks_after_reid - output_tracks_after_overlap_support),
        "trimmed_duplicate_tracks": int(output_tracks_after_overlap_support - output_tracks_after_trim),
        "trimmed_duplicate_frames": int(trim_debug.get("trimmed_frames", 0)),
        "contain_trimmed_duplicate_tracks": int(output_tracks_after_trim - output_tracks_after_contain_trim),
        "contain_trimmed_duplicate_frames": int(contain_trim_debug.get("trimmed_frames", 0)),
        "endpoint_trimmed_duplicate_tracks": int(output_tracks_after_contain_trim - output_tracks_after_endpoint_trim),
        "endpoint_trimmed_duplicate_frames": int(endpoint_trim_debug.get("trimmed_frames", 0)),
        "fragment_dropped_tracks": int(output_tracks_after_endpoint_trim - output_tracks_after_fragment_drop),
        "dropped_duplicate_tracks": int(output_tracks_after_fragment_drop - output_tracks_after_postmerge_repair),
        "temporal_link_merged_tracks": int(output_tracks_after_postmerge_repair - output_tracks_after_temporal_link),
        "edge_flicker_dropped_tracks": int(output_tracks_after_temporal_link - output_tracks_after_edge_flicker),
        "short_label_dropped_tracks": int(output_tracks_after_edge_flicker - output_tracks_after_short_label_drop),
        "gap_fill_merged_tracks": int(output_tracks_after_short_label_drop - output_tracks_after_gap_fill),
        "adjacent_flicker_merged_tracks": int(output_tracks_after_gap_fill - output_tracks_after_adjacent_flicker),
        "manual_merged_tracks": int(output_tracks_after_adjacent_flicker - len(output_tracks)),
        "merged_groups": int(merged_groups),
        "merge_pairs": int(len(decisions)),
        "max_gap": int(args.max_gap),
        "overlap_iou": float(args.overlap_iou),
        "gap_box_iou": float(args.gap_box_iou),
        "center_dist": float(args.center_dist),
        "label_counts_before": _counter_by_label(tracks),
        "label_counts_after": _counter_by_label(output_tracks),
        "short_thing_tracks_le3_before": _short_by_label(tracks),
        "short_thing_tracks_le3_after": _short_by_label(output_tracks),
        "decisions_preview": decisions[:200],
        "canonicalize_vehicle_labels": canonicalize_vehicle_debug,
        "proposal_drift_repair": proposal_drift_repair_debug,
        "proposal_miss_fallback": proposal_miss_fallback_debug,
        "reid_merge_tracks": reid_debug,
        "overlap_support_merge_tracks": overlap_support_debug,
        "trim_duplicate_frames": trim_debug,
        "trim_contained_duplicate_frames": contain_trim_debug,
        "trim_endpoint_duplicate_frames": endpoint_trim_debug,
        "drop_short_fragment_duplicate_tracks": fragment_drop_debug,
        "drop_short_duplicate_tracks": drop_debug,
        "postmerge_proposal_repair": postmerge_proposal_repair_debug,
        "temporal_link_tracks": temporal_link_debug,
        "drop_edge_flicker_tracks": edge_flicker_drop_debug,
        "drop_short_label_tracks": short_label_drop_debug,
        "gap_fill_short_tracks": gap_fill_debug,
        "adjacent_flicker_merge_tracks": adjacent_flicker_debug,
        "manual_merge_track_groups": manual_merge_debug,
    }
    merged.tracks = output_tracks
    merged.num_masklets = len(output_tracks)
    merged.debug["merge_sparse_thing_tracks"] = debug
    return merged, debug


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    needs_images = bool(int(args.render_video)) or bool(int(args.render_contact_sheet)) or bool(int(args.reid_merge_tracks))
    image_paths: Optional[List[str]]
    temp_dirs: List[str]
    if needs_images:
        image_paths, temp_dirs = _load_processing_frames(args, int(sparse.frame_height), int(sparse.frame_width), int(sparse.num_frames))
    else:
        image_paths, temp_dirs = None, []
        print("Skipped frame loading because render_video=0, render_contact_sheet=0, and reid_merge_tracks=0")
    merged, debug = merge_sparse(sparse, args, image_paths=image_paths)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_before_after.jpg"

    save_sparse_output(output_pt, merged)
    if int(args.render_video):
        if image_paths is None:
            raise RuntimeError("render_video=1 requires image paths")
        create_tracking_video_v2(
            image_paths,
            merged,
            str(output_video),
            fps=int(args.fps),
            mask_alpha=float(args.mask_alpha),
            render_style="clean",
        )
    if int(args.render_contact_sheet):
        if image_paths is None:
            raise RuntimeError("render_contact_sheet=1 requires image paths")
        make_contact_sheet(
            image_paths,
            sparse,
            merged,
            parse_contact_frames(args.contact_frames, merged.num_frames),
            contact_path,
            float(args.mask_alpha),
        )

    if int(args.fast_metrics):
        before_stats: Dict[str, Any] = {}
        after_stats: Dict[str, Any] = {}
        delta_stats: Dict[str, Any] = {}
        stats_after: List[Dict[str, Any]] = []
    else:
        before_stats = coverage_stats(sparse)
        after_stats = coverage_stats(merged)
        delta_stats = {
            key: float(after_stats[key]) - float(before_stats[key])
            for key in before_stats.keys()
            if key in after_stats
        }
        stats_after = track_stats(merged)

    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video) if int(args.render_video) else "",
        "contact_sheet": str(contact_path) if int(args.render_contact_sheet) else "",
        "fast_metrics": bool(int(args.fast_metrics)),
        "before": before_stats,
        "after": after_stats,
        "delta": delta_stats,
        "track_stats_after": stats_after,
        "merge_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
