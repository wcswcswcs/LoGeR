#!/usr/bin/env python3
"""
Stage C standalone runner -- Video Masklet Front-end with visualised output.

Supports two detector back-ends:
  * ``gdino``  – Grounding DINO
  * ``yoloe``  – YOLOE (ultralytics) with built-in segmentation

Supports two tracking back-ends:
  * ``sam2``   – SAM 2.1 video predictor
  * ``sam3``   – SAM 3 text-conditioned video predictor
  * ``sam31_multiplex`` – SAM 3.1 object-multiplex predictor

Usage examples::

    # --- Grounding DINO ---
    CUDA_VISIBLE_DEVICES=1 conda run -n loger python run_video_masklet_front_end.py \\
        --input data/examples/office \\
        --sam_backend sam2 \\
        --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \\
        --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
        --detector gdino \\
        --gdino_config Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
        --gdino_checkpoint /mnt/data/users/chengshun.wang/pjs/GroundingDINO/weights/groundingdino_swint_ogc.pth \\
        --output_video results/office_masklets.mp4

    # --- SAM3 + detector ---
    CUDA_VISIBLE_DEVICES=1 conda run -n loger python run_video_masklet_front_end.py \\
        --input data/examples/office \\
        --sam_backend sam3 \\
        --sam3_checkpoint ckpts/SAM3/sam3.pt \\
        --detector yoloe \\
        --yoloe_model yoloe-11l-seg.pt \\
        --output_video results/office_sam3.mp4

    # --- SAM3.1 multiplex + detector ---
    CUDA_VISIBLE_DEVICES=1 conda run -n loger python run_video_masklet_front_end.py \\
        --input data/examples/office \\
        --sam_backend sam31_multiplex \\
        --sam31_checkpoint ckpts/SAM3/sam3.1_multiplex.pt \\
        --detector yoloe \\
        --yoloe_model yoloe-11l-seg.pt \\
        --output_video results/office_sam31.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

GSAM2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grounded-SAM-2")
if GSAM2_ROOT not in sys.path:
    sys.path.insert(0, GSAM2_ROOT)

from loger.pipeline.video_masklet_frontend import (
    MaskletOutput,
    VideoMaskletFrontend,
    SEMANTIC_GROUP_NAMES,
)
from run_geometry_backbone_inference import collect_image_paths as collect_image_paths_geo


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (0, 128, 255), (128, 0, 255), (200, 200, 50), (50, 200, 200),
    (200, 50, 200), (100, 150, 200), (200, 100, 150), (150, 200, 100),
    (80, 80, 200), (200, 80, 80), (80, 200, 80), (180, 120, 60),
    (60, 180, 120), (120, 60, 180), (220, 180, 100), (100, 220, 180),
]


def get_colour(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Video Masklet Front-end (Stage C) verification.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True, help="Image folder or video file.")
    p.add_argument("--output_video", default="results/masklet_tracking.mp4")
    p.add_argument("--output_pt", default=None, help="Save tensors.")
    p.add_argument("--save_frames", default=None, help="Save annotated frames.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Frames per chunk (0 = all frames as one chunk).")
    p.add_argument("--chunk_overlap", type=int, default=4,
                   help="Overlap between consecutive chunks for cross-chunk ID matching.")

    # Tracking backend
    p.add_argument("--sam_backend", choices=["sam2", "sam3", "sam31_multiplex"], default="sam2")
    p.add_argument("--sam2_checkpoint", default=None)
    p.add_argument("--sam2_model_cfg", default=None)
    p.add_argument("--sam3_checkpoint", default=None)
    p.add_argument("--sam31_checkpoint", default=None,
                   help="Optional local SAM 3.1 multiplex checkpoint. If omitted, backend will try HuggingFace auto-download.")
    p.add_argument("--sam31_postprocess_batch_size", type=int, default=1,
                   help="SAM3.1 multiplex postprocess batch size. Smaller uses less GPU memory.")
    p.add_argument("--sam31_batched_grounding_batch_size", type=int, default=1,
                   help="SAM3.1 multiplex internal grounding batch size. Smaller uses less GPU memory.")
    p.add_argument("--sam31_offload_video_to_cpu", type=int, default=1,
                   help="SAM3.1 multiplex: keep decoded video frames on CPU during session inference (1/0).")
    p.add_argument("--sam31_offload_outputs_to_cpu", type=int, default=1,
                   help="SAM3.1 multiplex: offload detector outputs to CPU during eval to save GPU memory (1/0).")

    # Detector selection
    p.add_argument("--detector", choices=["gdino", "yoloe"], default="gdino")

    # Grounding DINO
    p.add_argument("--gdino_config", default=None)
    p.add_argument("--gdino_checkpoint", default=None)

    # YOLOE
    p.add_argument("--yoloe_model", default="yoloe-11l-seg.pt",
                   help="YOLOE model name or path (auto-downloaded if needed).")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Discovery
    p.add_argument("--thing_prompts", default=None, help="Comma-separated.")
    p.add_argument("--stuff_prompts", default=None, help="Comma-separated.")
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--ann_frame_idx", type=int, default=0,
                   help="Base discovery frame index inside each chunk.")
    p.add_argument("--discovery_frame_stride", type=int, default=8,
                   help="Run detector every N frames inside each chunk so late-appearing objects can be seeded.")
    p.add_argument("--max_thing_objects", type=int, default=15,
                   help="Max thing objects to track per chunk.")
    p.add_argument("--sam31_max_movable_objects", type=int, default=2,
                   help="For sam31_multiplex, max movable thing prompts tracked per chunk.")
    p.add_argument("--sam31_max_static_objects", type=int, default=1,
                   help="For sam31_multiplex, max static thing prompts tracked per chunk.")
    p.add_argument("--sam31_max_structure_objects", type=int, default=1,
                   help="For sam31_multiplex, max structure prompts tracked per chunk.")

    # Video
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mask_alpha", type=float, default=0.40)

    return p


def load_images(paths: list) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in paths])


def split_into_chunks(total_frames: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [(0, total_frames)]
    step = max(int(chunk_size) - int(overlap), 1)
    chunks: List[Tuple[int, int]] = []
    for start in range(0, total_frames, step):
        end = min(start + int(chunk_size), total_frames)
        chunks.append((start, end))
        if end == total_frames:
            break
    return chunks


def _labels_compatible(label_a: str, label_b: str) -> bool:
    a = label_a.strip().lower()
    b = label_b.strip().lower()
    return a == b or a in b or b in a


def _mask_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a.bool()
    b = mask_b.bool()
    inter = float((a & b).sum().item())
    union = float((a | b).sum().item())
    return inter / union if union > 0 else 0.0


def _mask_alignment_stats(mask_a: torch.Tensor, mask_b: torch.Tensor) -> Tuple[float, float, float]:
    a = mask_a.bool()
    b = mask_b.bool()
    inter = float((a & b).sum().item())
    area_a = float(a.sum().item())
    area_b = float(b.sum().item())
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    cover_a = inter / area_a if area_a > 0 else 0.0
    cover_b = inter / area_b if area_b > 0 else 0.0
    return iou, cover_a, cover_b


def _box_iou_xyxy_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a.tolist()]
    bx1, by1, bx2, by2 = [float(v) for v in box_b.tolist()]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _pack_mask_np(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool, copy=False)
    return np.packbits(mask_bool.reshape(-1))


def _unpack_mask_np(packed: np.ndarray, H: int, W: int) -> np.ndarray:
    flat = np.unpackbits(packed, count=H * W)
    return flat.reshape(H, W).astype(bool)


@dataclass
class SparseMaskletOutput:
    tracks: List[Dict]
    num_masklets: int
    num_frames: int
    frame_height: int
    frame_width: int
    debug: Dict[str, object] = field(default_factory=dict)


def _make_empty_track(total_frames: int, H: int, W: int, mo: MaskletOutput, j: int, birth_frame: int) -> Dict:
    return {
        "mask_by_frame": {},
        "box_by_frame": {},
        "q_by_frame": {},
        "area_by_frame": {},
        "L_sem": mo.L_sem[j] if j < len(mo.L_sem) else "?",
        "G_sem": int(mo.G_sem[j].item()),
        "W_sem": float(mo.W_sem[j].item()),
        "source_type": mo.source_type[j] if j < len(mo.source_type) else "?",
        "birth_frame": int(birth_frame),
        "frame_height": int(H),
        "frame_width": int(W),
    }


def _write_local_track_into_global(
    track: Dict,
    mo: MaskletOutput,
    local_idx: int,
    global_start: int,
    total_frames: int,
    preserve_overlap_until: int,
) -> None:
    for t_local in range(mo.num_frames):
        global_t = global_start + t_local
        if global_t >= total_frames:
            break
        if not bool(mo.V_mask[local_idx, t_local]):
            continue
        if global_t < preserve_overlap_until and global_t in track["mask_by_frame"]:
            continue
        mask_np = (mo.M_mask[local_idx, t_local].numpy() > 0.5)
        track["mask_by_frame"][global_t] = _pack_mask_np(mask_np)
        track["box_by_frame"][global_t] = mo.B_mask[local_idx, t_local].clone()
        track["q_by_frame"][global_t] = float(mo.Q_mask[local_idx, t_local].item())
        track["area_by_frame"][global_t] = float(mo.A_ratio[local_idx, t_local].item())


def _build_chunk_seed_detections(
    prev_mo: MaskletOutput,
    prev_start: int,
    curr_start: int,
    overlap: int,
) -> Dict[int, List[Dict]]:
    if overlap <= 0:
        return {}

    seed_dets: Dict[int, List[Dict]] = {}
    for j in range(prev_mo.num_masklets):
        source_type = prev_mo.source_type[j] if j < len(prev_mo.source_type) else "?"
        if source_type not in {"thing_tracked", "structure_tracked"}:
            continue

        chosen_local_t = None
        for local_curr_t in range(overlap):
            global_t = curr_start + local_curr_t
            prev_local_t = global_t - prev_start
            if not (0 <= prev_local_t < prev_mo.num_frames):
                continue
            if bool(prev_mo.V_mask[j, prev_local_t]):
                chosen_local_t = (local_curr_t, prev_local_t)
                break

        if chosen_local_t is None:
            continue

        local_curr_t, prev_local_t = chosen_local_t
        mask = prev_mo.M_mask[j, prev_local_t].numpy() > 0.5
        if mask.sum() <= 0:
            continue
        box = prev_mo.B_mask[j, prev_local_t].numpy().astype(np.float32)
        label = prev_mo.L_sem[j] if j < len(prev_mo.L_sem) else "unknown"
        sem_group = int(prev_mo.G_sem[j].item())
        conf = float(prev_mo.Q_mask[j, prev_local_t].item())
        area_ratio = float(prev_mo.A_ratio[j, prev_local_t].item())

        seed_dets.setdefault(local_curr_t, []).append({
            "mask": mask.astype(np.uint8),
            "box": box,
            "confidence": conf,
            "label": label,
            "raw_label": label,
            "sem_group": sem_group,
            "area_ratio": area_ratio,
            "is_seed_track": True,
        })

    return seed_dets


def build_chunk_discovery_indices(
    chunk_len: int,
    ann_frame_idx: int,
    discovery_frame_stride: int,
    overlap: int,
    chunk_idx: int,
) -> Optional[List[int]]:
    if chunk_len <= 0:
        return None
    if chunk_idx == 0 or overlap <= 0:
        return None

    start_idx = max(int(overlap), int(ann_frame_idx))
    if start_idx >= chunk_len:
        return [chunk_len - 1]

    indices = list(range(start_idx, chunk_len, max(int(discovery_frame_stride), 1)))
    if not indices:
        return [start_idx]
    return indices


def _compute_track_match_score(
    local_mo: MaskletOutput,
    local_idx: int,
    global_track: Dict,
    chunk_start: int,
    overlap: int,
) -> float:
    if overlap <= 0:
        return -1.0
    if global_track["source_type"] != (local_mo.source_type[local_idx] if local_idx < len(local_mo.source_type) else "?"):
        return -1.0
    local_group = int(local_mo.G_sem[local_idx].item())
    if global_track["G_sem"] != local_group:
        return -1.0
    local_label = local_mo.L_sem[local_idx] if local_idx < len(local_mo.L_sem) else "?"
    if not _labels_compatible(global_track["L_sem"], local_label):
        return -1.0

    overlap_len = min(overlap, local_mo.num_frames)
    local_union = None
    global_union = None
    frame_ious: List[float] = []
    frame_covers: List[float] = []
    frame_box_ious: List[float] = []
    frame_area_sims: List[float] = []
    local_vis = 0
    global_vis = 0

    for dt in range(overlap_len):
        global_t = chunk_start + dt
        local_visible = bool(local_mo.V_mask[local_idx, dt])
        global_visible = global_t in global_track["mask_by_frame"]
        if local_visible:
            local_mask = local_mo.M_mask[local_idx, dt].bool()
            local_union = local_mask if local_union is None else (local_union | local_mask)
            local_vis += 1
        if global_visible:
            global_mask = torch.from_numpy(
                _unpack_mask_np(
                    global_track["mask_by_frame"][global_t],
                    global_track["frame_height"],
                    global_track["frame_width"],
                )
            )
            global_union = global_mask if global_union is None else (global_union | global_mask)
            global_vis += 1
        if local_visible and global_visible:
            iou, cover_local, cover_global = _mask_alignment_stats(
                local_mo.M_mask[local_idx, dt],
                torch.from_numpy(
                    _unpack_mask_np(
                        global_track["mask_by_frame"][global_t],
                        global_track["frame_height"],
                        global_track["frame_width"],
                    )
                ),
            )
            frame_ious.append(iou)
            frame_covers.append(max(cover_local, cover_global))
            frame_box_ious.append(
                _box_iou_xyxy_torch(
                    local_mo.B_mask[local_idx, dt],
                    global_track["box_by_frame"][global_t],
                )
            )
            local_area = float(local_mo.A_ratio[local_idx, dt].item())
            global_area = float(global_track["area_by_frame"][global_t])
            if local_area > 0.0 and global_area > 0.0:
                frame_area_sims.append(min(local_area, global_area) / max(local_area, global_area))

    if local_vis == 0 or global_vis == 0:
        return -1.0

    union_iou = _mask_iou(local_union, global_union)
    mean_iou = sum(frame_ious) / len(frame_ious) if frame_ious else union_iou
    mean_cover = sum(frame_covers) / len(frame_covers) if frame_covers else 0.0
    mean_box_iou = sum(frame_box_ious) / len(frame_box_ious) if frame_box_ious else 0.0
    mean_area_sim = sum(frame_area_sims) / len(frame_area_sims) if frame_area_sims else 0.0

    if global_track["source_type"] == "stuff_static" and global_track["G_sem"] == 0:
        return (
            0.25 * mean_iou
            + 0.35 * mean_cover
            + 0.20 * union_iou
            + 0.10 * mean_box_iou
            + 0.10 * mean_area_sim
        )
    if frame_ious:
        return 0.7 * mean_iou + 0.3 * union_iou
    return union_iou


def _compute_structure_track_similarity(track_a: Dict, track_b: Dict, total_frames: int) -> float:
    if track_a["source_type"] != "stuff_static" or track_b["source_type"] != "stuff_static":
        return -1.0
    if track_a["G_sem"] != 0 or track_b["G_sem"] != 0:
        return -1.0
    if not _labels_compatible(track_a["L_sem"], track_b["L_sem"]):
        return -1.0

    frame_ious: List[float] = []
    frame_covers: List[float] = []
    frame_box_ious: List[float] = []
    frame_area_sims: List[float] = []
    overlap_frames = 0

    shared_frames = sorted(set(track_a["mask_by_frame"]).intersection(track_b["mask_by_frame"]))
    for t in shared_frames:
        overlap_frames += 1
        mask_a = torch.from_numpy(
            _unpack_mask_np(track_a["mask_by_frame"][t], track_a["frame_height"], track_a["frame_width"])
        )
        mask_b = torch.from_numpy(
            _unpack_mask_np(track_b["mask_by_frame"][t], track_b["frame_height"], track_b["frame_width"])
        )
        iou, cover_a, cover_b = _mask_alignment_stats(mask_a, mask_b)
        frame_ious.append(iou)
        frame_covers.append(max(cover_a, cover_b))
        frame_box_ious.append(_box_iou_xyxy_torch(track_a["box_by_frame"][t], track_b["box_by_frame"][t]))
        area_a = float(track_a["area_by_frame"][t])
        area_b = float(track_b["area_by_frame"][t])
        if area_a > 0.0 and area_b > 0.0:
            frame_area_sims.append(min(area_a, area_b) / max(area_a, area_b))

    if overlap_frames == 0:
        return -1.0

    mean_iou = sum(frame_ious) / len(frame_ious) if frame_ious else 0.0
    mean_cover = sum(frame_covers) / len(frame_covers) if frame_covers else 0.0
    mean_box_iou = sum(frame_box_ious) / len(frame_box_ious) if frame_box_ious else 0.0
    mean_area_sim = sum(frame_area_sims) / len(frame_area_sims) if frame_area_sims else 0.0
    return (
        0.20 * mean_iou
        + 0.45 * mean_cover
        + 0.20 * mean_box_iou
        + 0.15 * mean_area_sim
    )


def _merge_global_track_payload(primary: Dict, secondary: Dict, total_frames: int) -> None:
    primary["birth_frame"] = min(int(primary["birth_frame"]), int(secondary["birth_frame"]))
    primary["W_sem"] = max(float(primary["W_sem"]), float(secondary["W_sem"]))

    for t, packed_mask in secondary["mask_by_frame"].items():
        if t not in primary["mask_by_frame"]:
            primary["mask_by_frame"][t] = packed_mask
            primary["box_by_frame"][t] = secondary["box_by_frame"][t]
            primary["q_by_frame"][t] = secondary["q_by_frame"][t]
            primary["area_by_frame"][t] = secondary["area_by_frame"][t]
            continue
        if float(secondary["q_by_frame"][t]) > float(primary["q_by_frame"][t]):
            primary["mask_by_frame"][t] = packed_mask
            primary["box_by_frame"][t] = secondary["box_by_frame"][t]
            primary["q_by_frame"][t] = secondary["q_by_frame"][t]
            primary["area_by_frame"][t] = secondary["area_by_frame"][t]


def _deduplicate_structure_stuff_tracks(
    global_tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    removed = set()
    merge_debug: List[Dict[str, float]] = []

    while True:
        candidates: List[Tuple[float, int, int]] = []
        for i in range(len(global_tracks)):
            if i in removed:
                continue
            for j in range(i + 1, len(global_tracks)):
                if j in removed:
                    continue
                score = _compute_structure_track_similarity(global_tracks[i], global_tracks[j], total_frames)
                if score < 0.52:
                    continue
                vis_i = len(global_tracks[i]["mask_by_frame"])
                vis_j = len(global_tracks[j]["mask_by_frame"])
                mean_q_i = (
                    float(sum(global_tracks[i]["q_by_frame"].values()) / max(vis_i, 1))
                    if vis_i > 0 else 0.0
                )
                mean_q_j = (
                    float(sum(global_tracks[j]["q_by_frame"].values()) / max(vis_j, 1))
                    if vis_j > 0 else 0.0
                )
                if (vis_i, mean_q_i, -int(global_tracks[i]["birth_frame"])) >= (
                    vis_j, mean_q_j, -int(global_tracks[j]["birth_frame"])
                ):
                    primary, secondary = i, j
                else:
                    primary, secondary = j, i
                candidates.append((score, primary, secondary))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        merged_this_round = False
        used_this_round = set()
        for score, primary, secondary in candidates:
            if primary in removed or secondary in removed:
                continue
            if primary in used_this_round or secondary in used_this_round:
                continue
            _merge_global_track_payload(global_tracks[primary], global_tracks[secondary], total_frames)
            removed.add(secondary)
            used_this_round.add(primary)
            used_this_round.add(secondary)
            merge_debug.append({
                "score": float(score),
                "primary": int(primary),
                "secondary": int(secondary),
            })
            merged_this_round = True

        if not merged_this_round:
            break

    deduped = [track for idx, track in enumerate(global_tracks) if idx not in removed]
    return deduped, merge_debug


def merge_chunk_masklet_into_global(
    global_tracks: List[Dict],
    mo: MaskletOutput,
    chunk_idx: int,
    start: int,
    end: int,
    total_frames: int,
    chunk_overlap: int,
) -> Dict[str, object]:
    H = mo.frame_height
    W = mo.frame_width
    overlap = 0 if chunk_idx == 0 else min(int(chunk_overlap), mo.num_frames)
    local_to_global: Dict[int, int] = {}

    if chunk_idx > 0 and overlap > 0 and global_tracks:
        candidates: List[Tuple[float, int, int]] = []
        for j in range(mo.num_masklets):
            for g, track in enumerate(global_tracks):
                score = _compute_track_match_score(mo, j, track, start, overlap)
                if track["source_type"] == "thing_tracked":
                    threshold = 0.35
                elif track["source_type"] == "structure_tracked":
                    threshold = 0.16
                elif track["G_sem"] == 0:
                    threshold = 0.18
                else:
                    threshold = 0.30
                if score >= threshold:
                    candidates.append((score, j, g))
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_local = set()
        used_global = set()
        for score, j, g in candidates:
            if j in used_local or g in used_global:
                continue
            local_to_global[j] = g
            used_local.add(j)
            used_global.add(g)

    preserve_overlap_until = start + overlap
    for j in range(mo.num_masklets):
        global_idx = local_to_global.get(j)
        if global_idx is None:
            birth_frame = start + (mo.birth_frame[j] if j < len(mo.birth_frame) else 0)
            track = _make_empty_track(total_frames, H, W, mo, j, birth_frame)
            global_tracks.append(track)
            global_idx = len(global_tracks) - 1
        else:
            global_tracks[global_idx]["birth_frame"] = min(
                int(global_tracks[global_idx]["birth_frame"]),
                start + (mo.birth_frame[j] if j < len(mo.birth_frame) else 0),
            )

        _write_local_track_into_global(
            global_tracks[global_idx],
            mo,
            j,
            global_start=start,
            total_frames=total_frames,
            preserve_overlap_until=preserve_overlap_until if j in local_to_global else start,
        )

    return {
        "chunk_index": chunk_idx,
        "frame_range": (start, end),
        "matched_local_to_global": dict(local_to_global),
        "num_local_tracks": mo.num_masklets,
        "num_global_tracks_so_far": len(global_tracks),
    }


def finalize_global_tracks(
    global_tracks: List[Dict],
    total_frames: int,
    H: int,
    W: int,
    chunk_match_debug: List[Dict[str, object]],
) -> SparseMaskletOutput:
    if not global_tracks:
        return SparseMaskletOutput(
            tracks=[],
            num_masklets=0,
            num_frames=total_frames,
            frame_height=H,
            frame_width=W,
            debug={"chunk_match_debug": chunk_match_debug, "num_chunks": len(chunk_match_debug)},
        )

    global_tracks, structure_dedup_debug = _deduplicate_structure_stuff_tracks(
        global_tracks, total_frames,
    )
    return SparseMaskletOutput(
        tracks=global_tracks,
        num_masklets=len(global_tracks),
        num_frames=total_frames,
        frame_height=H,
        frame_width=W,
        debug={
            "chunk_match_debug": chunk_match_debug,
            "structure_dedup_debug": structure_dedup_debug,
            "num_chunks": len(chunk_match_debug),
            "J_thing": sum(1 for track in global_tracks if track["source_type"] == "thing_tracked"),
            "J_structure": sum(1 for track in global_tracks if track["source_type"] == "structure_tracked"),
            "J_stuff": sum(1 for track in global_tracks if track["source_type"] == "stuff_static"),
        },
    )


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def render_annotated_frame(
    rgb_np: np.ndarray,
    mo,
    frame_idx: int,
    mask_alpha: float = 0.40,
) -> np.ndarray:
    canvas = rgb_np.copy()
    overlay = canvas.copy()
    J = mo.num_masklets

    for j in range(J):
        if isinstance(mo, SparseMaskletOutput):
            track = mo.tracks[j]
            packed = track["mask_by_frame"].get(frame_idx)
            if packed is None:
                continue
            mask = _unpack_mask_np(packed, mo.frame_height, mo.frame_width)
            if mask.sum() == 0:
                continue
            box = track["box_by_frame"][frame_idx].numpy().astype(int)
            label = track["L_sem"]
            group_name = SEMANTIC_GROUP_NAMES.get(int(track["G_sem"]), "?")
            w_sem = float(track["W_sem"])
            q = float(track["q_by_frame"][frame_idx])
        else:
            if not mo.V_mask[j, frame_idx]:
                continue
            mask = mo.M_mask[j, frame_idx].numpy().astype(bool)
            if mask.sum() == 0:
                continue
            box = mo.B_mask[j, frame_idx].numpy().astype(int)
            label = mo.L_sem[j] if j < len(mo.L_sem) else "?"
            group_name = SEMANTIC_GROUP_NAMES.get(mo.G_sem[j].item(), "?")
            w_sem = mo.W_sem[j].item()
            q = mo.Q_mask[j, frame_idx].item()

        colour = get_colour(j)
        colour_np = np.array(colour, dtype=np.uint8)

        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1 - mask_alpha)
            + colour_np.astype(np.float32) * mask_alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, colour, 2)

        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        text = f"{label} [{group_name}] w={w_sem:.2f} q={q:.2f}"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(overlay, text, (x1 + 2, max(th + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    info = f"Frame {frame_idx}/{mo.num_frames}  Masklets: {J}"
    cv2.putText(overlay, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, info, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def create_tracking_video(
    image_paths: List[str],
    mo,
    output_path: str,
    fps: int = 10,
    mask_alpha: float = 0.40,
    save_frames_dir: Optional[str] = None,
) -> None:
    T = len(image_paths)
    H, W = mo.frame_height, mo.frame_width

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    for t in range(T):
        bgr_np = cv2.imread(image_paths[t], cv2.IMREAD_COLOR)
        if bgr_np is None:
            raise FileNotFoundError(f"Failed to read frame: {image_paths[t]}")
        rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
        if rgb_np.shape[0] != H or rgb_np.shape[1] != W:
            rgb_np = cv2.resize(rgb_np, (W, H))

        annotated = render_annotated_frame(rgb_np, mo, t, mask_alpha)
        frame_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

        if save_frames_dir:
            cv2.imwrite(os.path.join(save_frames_dir, f"frame_{t:05d}.jpg"), frame_bgr)

    writer.release()
    print(f"Saved tracking video to {output_path}  ({T} frames, {fps} FPS)")


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------
def print_masklet_output(mo) -> None:
    print("\n" + "=" * 72)
    print("MaskletOutput summary  (Stage C: Video Masklet Front-end)")
    print("=" * 72)
    print(f"  num_masklets         : {mo.num_masklets}")
    print(f"  num_frames           : {mo.num_frames}")
    print(f"  frame_size           : {mo.frame_height} x {mo.frame_width}")
    if isinstance(mo, SparseMaskletOutput):
        print("  storage             : sparse")
    else:
        print(f"  M_mask.shape         : {tuple(mo.M_mask.shape)}")

    jt = mo.debug.get("J_thing", "?")
    jsr = mo.debug.get("J_structure", 0)
    js = mo.debug.get("J_stuff", "?")
    print(f"  thing / structure / stuff : {jt} / {jsr} / {js}")

    if mo.num_masklets > 0:
        print()
        print(f"  {'ID':>3s}  {'Type':14s}  {'Label':20s}  {'Group':22s}  {'W_sem':>5s}  "
              f"{'Birth':>5s}  {'Visible':>7s}  {'MeanQ':>5s}  {'MeanArea':>8s}")
        print("  " + "-" * 115)
        for j in range(mo.num_masklets):
            if isinstance(mo, SparseMaskletOutput):
                track = mo.tracks[j]
                vis = len(track["mask_by_frame"])
                mq = sum(track["q_by_frame"].values()) / max(vis, 1) if vis > 0 else 0
                ma = sum(track["area_by_frame"].values()) / max(vis, 1) if vis > 0 else 0
                g = int(track["G_sem"])
                gn = SEMANTIC_GROUP_NAMES.get(g, "?")
                lbl = track["L_sem"]
                bf = int(track["birth_frame"])
                stype = track["source_type"]
                w_sem = float(track["W_sem"])
            else:
                vis = mo.V_mask[j].sum().item()
                mq = mo.Q_mask[j, mo.V_mask[j]].mean().item() if vis > 0 else 0
                ma = mo.A_ratio[j, mo.V_mask[j]].mean().item() if vis > 0 else 0
                g = mo.G_sem[j].item()
                gn = SEMANTIC_GROUP_NAMES.get(g, "?")
                lbl = mo.L_sem[j] if j < len(mo.L_sem) else "?"
                bf = mo.birth_frame[j] if j < len(mo.birth_frame) else -1
                stype = mo.source_type[j] if j < len(mo.source_type) else "?"
                w_sem = mo.W_sem[j].item()
            print(f"  {j:3d}  {stype:14s}  {lbl:20s}  {gn:22s}  {w_sem:5.2f}  "
                  f"{bf:5d}  {vis:5.0f}/{mo.num_frames}  {mq:5.3f}  {ma:8.5f}")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if args.sam_backend == "sam2":
        if not args.sam2_checkpoint or not args.sam2_model_cfg:
            sys.exit("--sam2_checkpoint and --sam2_model_cfg are required when --sam_backend sam2")
    elif args.sam_backend == "sam3":
        if not args.sam3_checkpoint:
            sys.exit("--sam3_checkpoint is required when --sam_backend sam3")
    elif args.sam_backend == "sam31_multiplex":
        if args.sam31_checkpoint and not os.path.exists(args.sam31_checkpoint):
            sys.exit(f"--sam31_checkpoint not found: {args.sam31_checkpoint}")

    if args.detector == "gdino":
        if not args.gdino_config or not args.gdino_checkpoint:
            sys.exit("--gdino_config and --gdino_checkpoint are required when --detector gdino")

    temp_dir = None
    try:
        image_paths, temp_dir = collect_image_paths_geo(
            args.input, args.start_frame, args.end_frame, args.stride,
        )
        if not image_paths:
            sys.exit("No images found.")
        print(f"Collected {len(image_paths)} images.")
        total_frames = int(len(image_paths))
        sample_img = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
        if sample_img is None:
            sys.exit(f"Failed to read first frame: {image_paths[0]}")
        sample_h, sample_w = sample_img.shape[:2]
        print(f"Frame size: ({sample_h}, {sample_w})  total_frames={total_frames}")

        chunks = split_into_chunks(total_frames, args.chunk_size, args.chunk_overlap)
        print(
            f"Chunk schedule: {len(chunks)} chunk(s)  "
            f"size={args.chunk_size or total_frames}  overlap={args.chunk_overlap}"
        )

        # Build kwargs
        frontend_kwargs: dict = dict(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            ann_frame_idx=args.ann_frame_idx,
            discovery_frame_stride=args.discovery_frame_stride,
            max_thing_objects=args.max_thing_objects,
            sam31_max_movable_objects=args.sam31_max_movable_objects,
            sam31_max_static_objects=args.sam31_max_static_objects,
            sam31_max_structure_objects=args.sam31_max_structure_objects,
        )
        if args.thing_prompts:
            frontend_kwargs["thing_prompts"] = [s.strip() for s in args.thing_prompts.split(",")]
        if args.stuff_prompts:
            frontend_kwargs["stuff_prompts"] = [s.strip() for s in args.stuff_prompts.split(",")]

        # Build frontend
        print(f"Loading models (sam_backend={args.sam_backend}, detector={args.detector}) ...")
        t0 = time.time()

        build_kwargs: dict = dict(
            sam_backend=args.sam_backend,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
            sam3_checkpoint=args.sam3_checkpoint,
            sam31_checkpoint=args.sam31_checkpoint,
            sam31_postprocess_batch_size=args.sam31_postprocess_batch_size,
            sam31_batched_grounding_batch_size=args.sam31_batched_grounding_batch_size,
            sam31_offload_video_to_cpu=args.sam31_offload_video_to_cpu,
            sam31_offload_outputs_to_cpu=args.sam31_offload_outputs_to_cpu,
            device=args.device,
            detector_type=args.detector,
        )
        if args.detector == "gdino":
            build_kwargs["gdino_config"] = args.gdino_config
            build_kwargs["gdino_checkpoint"] = args.gdino_checkpoint
        elif args.detector == "yoloe":
            build_kwargs["yoloe_model"] = args.yoloe_model

        frontend = VideoMaskletFrontend.from_config(**build_kwargs, **frontend_kwargs)
        print(f"Models loaded in {time.time() - t0:.1f}s")

        # Run Stage C chunk-by-chunk
        print("Running Video Masklet Front-end (Stage C) ...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        global_tracks: List[Dict] = []
        chunk_match_debug: List[Dict[str, object]] = []
        prev_chunk_output: Optional[MaskletOutput] = None
        prev_chunk_start: Optional[int] = None
        for ci, (start, end) in enumerate(chunks):
            seed_detections_by_frame = None
            discovery_frame_indices = build_chunk_discovery_indices(
                chunk_len=end - start,
                ann_frame_idx=args.ann_frame_idx,
                discovery_frame_stride=args.discovery_frame_stride,
                overlap=args.chunk_overlap,
                chunk_idx=ci,
            )
            if (
                prev_chunk_output is not None
                and prev_chunk_start is not None
                and args.chunk_overlap > 0
            ):
                seed_detections_by_frame = _build_chunk_seed_detections(
                    prev_chunk_output,
                    prev_chunk_start,
                    start,
                    args.chunk_overlap,
                )
            print(f"\n{'#' * 72}")
            print(f"# Chunk {ci}/{len(chunks) - 1}  frames [{start}, {end})")
            print(f"{'#' * 72}")
            chunk_t0 = time.time()
            chunk_output = frontend.run_from_paths(
                image_paths[start:end],
                discovery_frame_indices=discovery_frame_indices,
                seed_detections_by_frame=seed_detections_by_frame,
            )
            print(f"  Stage C chunk done in {time.time() - chunk_t0:.2f}s")
            if chunk_output.debug.get("discovery_frames") is not None:
                print(f"  Discovery frames (local): {chunk_output.debug['discovery_frames']}")
            print_masklet_output(chunk_output)
            chunk_match_debug.append(
                merge_chunk_masklet_into_global(
                    global_tracks,
                    chunk_output,
                    chunk_idx=ci,
                    start=start,
                    end=end,
                    total_frames=total_frames,
                    chunk_overlap=args.chunk_overlap,
                )
            )
            prev_chunk_output = chunk_output
            prev_chunk_start = start
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        masklet_output = finalize_global_tracks(
            global_tracks,
            total_frames=total_frames,
            H=sample_h,
            W=sample_w,
            chunk_match_debug=chunk_match_debug,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"\nStage C total done in {time.time() - t0:.2f}s")

        print_masklet_output(masklet_output)

        create_tracking_video(
            image_paths, masklet_output, args.output_video,
            fps=args.fps, mask_alpha=args.mask_alpha,
            save_frames_dir=args.save_frames,
        )

        if args.output_pt:
            os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
            sparse_tracks = []
            for track in masklet_output.tracks:
                frames = sorted(track["mask_by_frame"].keys())
                sparse_tracks.append({
                    "L_sem": track["L_sem"],
                    "G_sem": int(track["G_sem"]),
                    "W_sem": float(track["W_sem"]),
                    "source_type": track["source_type"],
                    "birth_frame": int(track["birth_frame"]),
                    "frames": frames,
                    "packed_masks": [
                        torch.from_numpy(track["mask_by_frame"][t].copy()) for t in frames
                    ],
                    "boxes": torch.stack([track["box_by_frame"][t] for t in frames], dim=0)
                    if frames else torch.zeros(0, 4, dtype=torch.float32),
                    "scores": torch.tensor([track["q_by_frame"][t] for t in frames], dtype=torch.float32),
                    "area_ratio": torch.tensor([track["area_by_frame"][t] for t in frames], dtype=torch.float32),
                })
            torch.save({
                "format": "sparse_masklets_v1",
                "frame_height": masklet_output.frame_height,
                "frame_width": masklet_output.frame_width,
                "num_masklets": masklet_output.num_masklets,
                "num_frames": masklet_output.num_frames,
                "tracks": sparse_tracks,
                "debug": masklet_output.debug,
            }, args.output_pt)
            print(f"Saved masklet tensors to {args.output_pt}")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
