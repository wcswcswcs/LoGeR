#!/usr/bin/env python3
"""
Stage C standalone runner -- EfficientSAM3 + YOLOE video masklet front-end.

This runner mirrors ``run_video_masklet_front_end.py`` but swaps the SAM3
predictor for the EfficientSAM3 video predictor from
``third_party/efficientsam3``.

Usage example::

    CUDA_VISIBLE_DEVICES=0 conda run -n loger python run_efficient_video_masklet_front_end.py \\
        --input data/examples/taylor.mp4 \\
        --yoloe_model yoloe-11l-seg.pt \\
        --output_video results/taylor_efficientsam3.mp4
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _requested_tracker_backend_from_argv() -> Optional[str]:
    for idx, arg in enumerate(sys.argv):
        if arg == "--tracker_backend" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1].strip().lower()
        if arg.startswith("--tracker_backend="):
            return arg.split("=", 1)[1].strip().lower()
    return None


_REQUESTED_TRACKER_BACKEND = _requested_tracker_backend_from_argv()
_DEFAULT_GSAM2_ROOT = (
    os.path.join(REPO_ROOT, "third_party", "EdgeTAM")
    if _REQUESTED_TRACKER_BACKEND in {"edgetam", "edge_tam"}
    else os.path.join(REPO_ROOT, "Grounded-SAM-2")
)
GSAM2_ROOT = os.environ.get("GSAM2_ROOT", _DEFAULT_GSAM2_ROOT)
os.environ.setdefault("GSAM2_ROOT", GSAM2_ROOT)
if GSAM2_ROOT not in sys.path:
    sys.path.insert(0, GSAM2_ROOT)

from loger.pipeline.efficient_video_masklet_frontend import (
    DEFAULT_EFFICIENTSAM3_FILENAME,
    DEFAULT_EFFICIENTSAM3_REPO_ID,
    DEFAULT_EDGETAM_CHECKPOINT,
    DEFAULT_EDGETAM_MODEL_CFG,
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_MODEL_CFG,
    DEFAULT_SAM3_CHECKPOINT,
    EfficientVideoMaskletFrontend,
    MaskletOutput,
    SEMANTIC_GROUP_MOVABLE_THING,
    SEMANTIC_GROUP_NAMES,
    SEMANTIC_GROUP_STATIC_THING,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR,
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

DEFAULT_EFFICIENT_THING_PROMPTS = "person,people,man,woman,singer,dancer,guitar,musical instrument,monitor"
DEFAULT_EFFICIENT_STUFF_PROMPTS = "floor,wall,ceiling"


def get_colour(idx: int) -> tuple:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EfficientSAM3 + YOLOE Video Masklet Front-end (Stage C).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True, help="Image folder or video file.")
    p.add_argument("--output_video", default="results/efficient_masklet_tracking.mp4")
    p.add_argument("--output_pt", default=None, help="Save tensors.")
    p.add_argument("--save_frames", default=None, help="Save annotated frames.")
    p.add_argument("--checkpoint_pt", default=None,
                   help="Chunk-level resume checkpoint. Defaults to <output_pt>.checkpoint.pt when --output_pt is set.")
    p.add_argument("--resume_checkpoint", action="store_true",
                   help="Resume from --checkpoint_pt if it exists.")
    p.add_argument("--checkpoint_every_chunks", type=int, default=1,
                   help="Save a resume checkpoint every N chunks; 0 disables checkpointing.")
    p.add_argument("--print_chunk_summaries", type=int, default=0,
                   help="Print full per-chunk masklet tables. Default prints compact one-line summaries.")
    p.add_argument("--max_chunks_this_run", type=int, default=0,
                   help="If >0, process at most this many chunks, save a checkpoint, and exit before finalization.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=64,
                   help="Frames per chunk (0 = all frames as one chunk).")
    p.add_argument("--chunk_overlap", type=int, default=4,
                   help="Overlap between consecutive chunks for cross-chunk ID matching.")

    # SAM3 + light tracker / EfficientSAM3 compatibility
    p.add_argument("--tracker_backend", default="sam2",
                   choices=["sam2", "edgetam", "cutie", "efficientsam3", "efficient"],
                   help="Video propagation backend. Default uses SAM3+YOLOE discovery + SAM2.1 tracking.")
    p.add_argument("--sam3_checkpoint", default=DEFAULT_SAM3_CHECKPOINT,
                   help="SAM3 checkpoint used for keyframe prompt segmentation in --tracker_backend cutie.")
    p.add_argument("--sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT,
                   help="SAM2.1 checkpoint used by --tracker_backend sam2.")
    p.add_argument("--sam2_model_cfg", default=DEFAULT_SAM2_MODEL_CFG,
                   help="SAM2.1 model config used by --tracker_backend sam2.")
    p.add_argument("--edgetam_checkpoint", default=DEFAULT_EDGETAM_CHECKPOINT,
                   help="EdgeTAM checkpoint used by --tracker_backend edgetam.")
    p.add_argument("--edgetam_model_cfg", default=DEFAULT_EDGETAM_MODEL_CFG,
                   help="EdgeTAM model config used by --tracker_backend edgetam.")
    p.add_argument("--cutie_max_internal_size", type=int, default=480,
                   help="Cutie internal short-edge size. Lower is faster; -1 keeps original size.")
    p.add_argument("--sam3_cutie_sam_confidence_threshold", type=float, default=0.10,
                   help="SAM3 image-prompt confidence threshold before local filtering.")
    p.add_argument("--sam3_cutie_detection_frame_count", type=int, default=6,
                   help="Number of SAM3 prompt-discovery frames per chunk for the Cutie backend.")
    p.add_argument("--sam3_cutie_max_prompt_dets_per_label", type=int, default=4,
                   help="Max SAM3 masks kept per label per discovery frame.")
    p.add_argument("--sam3_cutie_use_yoloe", type=int, default=1,
                   help="Also add YOLOE detections as seeds for the SAM3+Cutie backend.")

    # EfficientSAM3 legacy backend
    p.add_argument("--efficientsam3_checkpoint", default=None,
                   help="Local merged EfficientSAM3 checkpoint. If omitted, auto-downloads a default model.")
    p.add_argument("--efficientsam3_repo_id", default=DEFAULT_EFFICIENTSAM3_REPO_ID,
                   help="Hugging Face repo used when auto-downloading EfficientSAM3.")
    p.add_argument("--efficientsam3_filename", default=DEFAULT_EFFICIENTSAM3_FILENAME,
                   help="Filename inside the HF repo used when auto-downloading EfficientSAM3.")
    p.add_argument("--efficientsam3_cache_dir", default="ckpts/EfficientSAM3",
                   help="Local cache dir for EfficientSAM3 downloads.")
    p.add_argument("--efficientsam3_backbone_type", default="efficientvit",
                   choices=["repvit", "tinyvit", "efficientvit"],
                   help="EfficientSAM3 student vision backbone family.")
    p.add_argument("--efficientsam3_model_name", default="b0",
                   help="Model variant inside the chosen backbone family, e.g. b0 / 11m / m1.1.")
    p.add_argument("--efficientsam3_text_encoder_type", default=None,
                   help="Optional student text encoder type, e.g. MobileCLIP-S0.")
    p.add_argument("--efficientsam3_text_context_length", type=int, default=77,
                   help="Context length when using a student text encoder.")

    # YOLOE
    p.add_argument("--yoloe_model", default="yoloe-11l-seg.pt",
                   help="YOLOE model name or path (auto-downloaded if needed).")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Discovery
    p.add_argument("--thing_prompts", default=DEFAULT_EFFICIENT_THING_PROMPTS, help="Comma-separated.")
    p.add_argument("--stuff_prompts", default=DEFAULT_EFFICIENT_STUFF_PROMPTS, help="Comma-separated.")
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--ann_frame_idx", type=int, default=0,
                   help="Base discovery frame index inside each chunk.")
    p.add_argument("--discovery_frame_stride", type=int, default=4,
                   help="Run detector every N frames inside each chunk so late-appearing objects can be seeded.")
    p.add_argument("--max_thing_objects", type=int, default=24,
                   help="Max thing objects to track per chunk.")
    p.add_argument("--sam31_text_track_labels", default="",
                   help="Comma-separated labels for text tracking. Empty by default because the local EfficientSAM3 checkpoint is object-prompt oriented.")
    p.add_argument("--sam31_structure_prompt_labels", default="wall,floor,ceiling",
                   help="Comma-separated structure labels directly queried with EfficientSAM3 text prompts.")
    p.add_argument("--sam31_structure_prompt_frame_count", type=int, default=1,
                   help="Number of discovery frames per chunk used for direct structure prompt detection.")
    p.add_argument("--sam31_structure_prompt_chunk_stride", type=int, default=1,
                   help="Run direct structure prompts every N chunks; 1 = every chunk, 0 = disabled.")
    p.add_argument("--sam31_person_refresh_prompt_frames", type=int, default=1,
                   help="Extra late-frame text prompt refreshes for person/people per chunk.")
    p.add_argument("--sam31_nontext_object_prompt_budget", type=int, default=0,
                   help="Max non-text YOLOE candidates to propagate with object prompts per chunk.")
    p.add_argument("--sam31_nontext_object_prompt_min_support", type=int, default=2,
                   help="Minimum multi-frame detector support for non-seed non-text object prompts.")
    p.add_argument("--sam31_nontext_sparse_support", type=int, default=1,
                   help="Keep non-text YOLOE thing masks as sparse detector support without video propagation.")
    p.add_argument("--sam31_max_text_prompt_objects", type=int, default=0,
                   help="Cap text-prompt objects propagated per text query; 0 keeps all.")
    p.add_argument("--sam31_max_movable_objects", type=int, default=8,
                   help="Max movable thing prompt tracks per chunk.")
    p.add_argument("--sam31_max_static_objects", type=int, default=2,
                   help="Max static thing prompt tracks per chunk.")
    p.add_argument("--sam31_max_structure_objects", type=int, default=3,
                   help="Max tracked structure prompt objects per chunk.")
    p.add_argument("--segformer_stuff_model", default="nvidia/segformer-b0-finetuned-ade-512-512",
                   help="Optional ADE20K semantic segmentation model for STUFF. Empty string disables it.")
    p.add_argument("--segformer_stuff_stride", type=int, default=10,
                   help="Run SegFormer STUFF every N frames and reuse masks between samples; 0 disables it.")
    p.add_argument("--segformer_stuff_batch_size", type=int, default=8,
                   help="Batch size for SegFormer STUFF inference.")
    p.add_argument("--segformer_stuff_labels", default="floor,ceiling,stair",
                   help="Comma-separated STUFF labels to emit from ADE20K SegFormer.")
    p.add_argument("--segformer_stuff_replace_existing", type=int, default=1,
                   help="Replace SAM/YOLOE structure tracks with SegFormer STUFF tracks.")

    # Video
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mask_alpha", type=float, default=0.40)

    return p


def load_images(paths: list) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in paths])


def _run_with_heartbeat(label: str, fn, interval_s: float = 5.0):
    done = threading.Event()

    def _beat() -> None:
        start = time.time()
        while not done.wait(interval_s):
            print(f"{label} ... {time.time() - start:.1f}s", flush=True)

    thread = threading.Thread(target=_beat, daemon=True)
    thread.start()
    try:
        return fn()
    finally:
        done.set()
        thread.join(timeout=0.2)


def split_into_chunks(total_frames: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [(0, total_frames)]
    step = max(int(chunk_size) - int(overlap), 1)
    min_tail = max(int(overlap) + 1, int(chunk_size) // 2)
    chunks: List[Tuple[int, int]] = []
    for start in range(0, total_frames, step):
        end = min(start + int(chunk_size), total_frames)
        if 0 < total_frames - end <= min_tail:
            end = total_frames
        chunks.append((start, end))
        if end == total_frames:
            break
    return chunks


_HUMAN_LABELS = {
    "person",
    "people",
    "man",
    "woman",
    "singer",
    "dancer",
    "performer",
    "human",
    "human figure",
    "rider",
}


def _labels_compatible(label_a: str, label_b: str) -> bool:
    a = label_a.strip().lower()
    b = label_b.strip().lower()
    if a in _HUMAN_LABELS and b in _HUMAN_LABELS:
        return True
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


def _track_label_is_person(track: Dict) -> bool:
    label = str(track.get("L_sem", ""))
    return _labels_compatible(label, "person") or _labels_compatible(label, "people")


def _should_replace_track_frame(primary: Dict, secondary: Dict, frame_idx: int) -> bool:
    primary_q = float(primary.get("q_by_frame", {}).get(frame_idx, 0.0))
    secondary_q = float(secondary.get("q_by_frame", {}).get(frame_idx, 0.0))
    if not (_track_label_is_person(primary) and _track_label_is_person(secondary)):
        return secondary_q > primary_q

    primary_area = float(primary.get("area_by_frame", {}).get(frame_idx, 0.0))
    secondary_area = float(secondary.get("area_by_frame", {}).get(frame_idx, 0.0))
    if secondary_area <= 0.0:
        return False
    if primary_area <= 0.0:
        return True

    # EfficientSAM3 often assigns perfect confidence to a guitar/torso shard.
    # For person tracks, prefer the more complete silhouette unless the smaller
    # mask is substantially more confident.
    if secondary_area >= 1.22 * primary_area and secondary_q >= primary_q - 0.18:
        return True
    if secondary_area >= 0.82 * primary_area and secondary_q > primary_q + 0.03:
        return True
    if secondary_area < 0.62 * primary_area and primary_area >= 0.018:
        return False
    if secondary_area < 0.45 * primary_area:
        return False
    return secondary_q > primary_q + 0.08


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
        if _should_replace_track_frame(primary, secondary, int(t)):
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

    def _render_priority(j: int) -> Tuple[int, int]:
        if isinstance(mo, SparseMaskletOutput):
            track = mo.tracks[j]
            label = str(track.get("L_sem", "")).strip().lower()
            source_type = str(track.get("source_type", ""))
            sem_group = int(track.get("G_sem", -1))
        else:
            label = str(mo.L_sem[j] if j < len(mo.L_sem) else "").strip().lower()
            source_type = str(mo.source_type[j] if j < len(mo.source_type) else "")
            sem_group = int(mo.G_sem[j].item())
        if source_type in {"structure_tracked", "stuff_static"}:
            return 0, j
        if sem_group == SEMANTIC_GROUP_STATIC_THING:
            return 1, j
        if label in {"person", "people"}:
            return 3, j
        return 2, j

    for j in sorted(range(J), key=_render_priority):
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
        if t > 0 and (t % 120 == 0 or t == T - 1):
            print(f"  Rendering video frame {t}/{T}", flush=True)
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


from run_video_masklet_front_end import (  # noqa: E402
    SparseMaskletOutput as _MatureSparseMaskletOutput,
    _pack_mask_np as _mature_pack_mask_np,
    _unpack_mask_np as _mature_unpack_mask_np,
    build_chunk_discovery_indices as _mature_build_chunk_discovery_indices,
    finalize_global_tracks as _mature_finalize_global_tracks,
    merge_chunk_masklet_into_global as _mature_merge_chunk_masklet_into_global,
)

SparseMaskletOutput = _MatureSparseMaskletOutput
_pack_mask_np = _mature_pack_mask_np
_unpack_mask_np = _mature_unpack_mask_np
merge_chunk_masklet_into_global = _mature_merge_chunk_masklet_into_global


def _build_global_chunk_seed_detections(
    global_tracks: List[Dict],
    curr_start: int,
    overlap: int,
) -> Dict[int, List[Dict]]:
    """Carry efficient-path seeds across chunks, including STUFF anchors.

    The mature SAM3.1 path only reseeds tracked things/structures because SAM
    handles temporal propagation.  EfficientSAM3 keeps STUFF on the cheaper
    detector/flow path, so a propagated floor/wall mask must also be available
    in the next chunk's overlap frames; otherwise structure regions blink out
    after one chunk.
    """
    if overlap <= 0:
        return {}

    candidates_by_frame: Dict[int, List[Dict]] = {}
    allowed_sources = {"thing_tracked", "structure_tracked", "stuff_static"}
    for global_idx, track in enumerate(global_tracks):
        source_type = str(track.get("source_type", "?"))
        if source_type not in allowed_sources:
            continue
        if (
            source_type == "thing_tracked"
            and int(track.get("G_sem", -1)) == SEMANTIC_GROUP_STATIC_THING
        ):
            continue

        for local_curr_t in range(overlap):
            global_t = curr_start + local_curr_t
            if global_t not in track["mask_by_frame"]:
                continue

            mask = _unpack_mask_np(
                track["mask_by_frame"][global_t],
                int(track["frame_height"]),
                int(track["frame_width"]),
            )
            if mask.sum() <= 0:
                continue
            box = track["box_by_frame"][global_t].numpy().astype(np.float32)
            label = str(track.get("L_sem", "unknown"))
            if label.strip().lower() == "book":
                continue
            sem_group = int(track.get("G_sem", -1))
            conf = float(track["q_by_frame"][global_t])
            area_ratio = float(track["area_by_frame"][global_t])

            candidates_by_frame.setdefault(local_curr_t, []).append({
                "mask": mask.astype(np.uint8),
                "box": box,
                "confidence": conf,
                "label": label,
                "raw_label": label,
                "sem_group": sem_group,
                "area_ratio": area_ratio,
                "is_seed_track": True,
                "seed_global_track_idx": int(global_idx),
                "source_type": source_type,
            })

    # Carrying every historical STUFF/person shard into EdgeTAM makes later
    # chunks slow and can kill the process. Keep continuity seeds, but cap them
    # to the strongest active instances in each overlap frame.
    seed_dets: Dict[int, List[Dict]] = {}
    stuff_priority = {"floor": 0, "wall": 1, "ceiling": 2, "curtain": 3, "screen": 4}
    for local_t, candidates in candidates_by_frame.items():
        thing_candidates: List[Dict] = []
        stuff_best_by_label: Dict[str, Dict] = {}
        for det in candidates:
            label_l = str(det.get("label", "")).strip().lower()
            source_type = str(det.get("source_type", ""))
            area_ratio = float(det.get("area_ratio", 0.0))
            conf = float(det.get("confidence", 0.0))
            if source_type == "thing_tracked":
                if label_l in {"person", "people"} and area_ratio < 0.010:
                    continue
                thing_candidates.append(det)
                continue

            if source_type in {"structure_tracked", "stuff_static"}:
                if label_l not in stuff_priority:
                    continue
                score = conf + 2.0 * min(max(area_ratio, 0.0), 0.30)
                prev = stuff_best_by_label.get(label_l)
                if prev is None:
                    stuff_best_by_label[label_l] = det
                else:
                    prev_score = float(prev.get("confidence", 0.0)) + 2.0 * min(
                        max(float(prev.get("area_ratio", 0.0)), 0.0),
                        0.30,
                    )
                    if score > prev_score:
                        stuff_best_by_label[label_l] = det

        thing_candidates.sort(
            key=lambda det: (
                0 if str(det.get("label", "")).strip().lower() in {"person", "people"} else 1,
                -float(det.get("area_ratio", 0.0)),
                -float(det.get("confidence", 0.0)),
            )
        )
        kept_things: List[Dict] = []
        per_label_counts: Dict[str, int] = {}
        for det in thing_candidates:
            label_l = str(det.get("label", "")).strip().lower()
            label_budget = 3 if label_l in {"person", "people"} else 1
            if per_label_counts.get(label_l, 0) >= label_budget:
                continue
            if len(kept_things) >= 5:
                break
            kept_things.append(det)
            per_label_counts[label_l] = per_label_counts.get(label_l, 0) + 1

        kept_stuff = [
            stuff_best_by_label[label]
            for label in sorted(stuff_best_by_label, key=lambda name: stuff_priority.get(name, 99))
        ][:5]
        kept = kept_things + kept_stuff
        for det in kept:
            det.pop("source_type", None)
        if kept:
            seed_dets[int(local_t)] = kept

    return seed_dets


def build_chunk_discovery_indices(
    chunk_len: int,
    ann_frame_idx: int,
    discovery_frame_stride: int,
    overlap: int,
    chunk_idx: int,
) -> Optional[List[int]]:
    base = _mature_build_chunk_discovery_indices(
        chunk_len=chunk_len,
        ann_frame_idx=ann_frame_idx,
        discovery_frame_stride=discovery_frame_stride,
        overlap=overlap,
        chunk_idx=chunk_idx,
    )
    if chunk_idx <= 0 or overlap <= 0 or chunk_len <= 0:
        return base

    # EfficientSAM3 can drift badly if a stale person seed is propagated into
    # the next chunk. Running detection on the overlap frames gives the
    # frontend a cheap sanity check before carrying those seeds forward.
    overlap_indices = list(range(0, min(int(overlap), int(chunk_len))))
    if base is None:
        return overlap_indices
    return sorted(set(int(v) for v in base).union(overlap_indices))


def _local_masklet_stats(mo: MaskletOutput, local_idx: int) -> Dict[str, float]:
    visible_mask = mo.V_mask[local_idx].bool()
    visible = int(visible_mask.sum().item())
    if visible <= 0:
        return {"visible": 0.0, "mean_area": 0.0, "max_area": 0.0}
    areas = mo.A_ratio[local_idx, visible_mask].float()
    return {
        "visible": float(visible),
        "mean_area": float(areas.mean().item()),
        "max_area": float(areas.max().item()),
    }


def _compact_chunk_output_before_global_merge(
    mo: MaskletOutput,
) -> Tuple[MaskletOutput, List[Dict[str, object]]]:
    if mo.num_masklets <= 0:
        return mo, []

    stats_by_idx = [_local_masklet_stats(mo, j) for j in range(mo.num_masklets)]
    person_indices = [
        j for j in range(mo.num_masklets)
        if _is_person_track({
            "L_sem": mo.L_sem[j] if j < len(mo.L_sem) else "",
        })
    ]
    dominant_person_idx: Optional[int] = None
    if person_indices:
        dominant_person_idx = max(
            person_indices,
            key=lambda j: (
                stats_by_idx[j]["visible"] * max(stats_by_idx[j]["mean_area"], 1e-4) ** 0.5,
                stats_by_idx[j]["mean_area"],
                stats_by_idx[j]["max_area"],
            ),
        )

    keep: List[int] = []
    pruned: List[Dict[str, object]] = []
    T = max(int(mo.num_frames), 1)
    for j in range(mo.num_masklets):
        label_l = str(mo.L_sem[j] if j < len(mo.L_sem) else "").strip().lower()
        source_type = str(mo.source_type[j] if j < len(mo.source_type) else "")
        sem_group = int(mo.G_sem[j].item())
        stats = stats_by_idx[j]
        visible = int(stats["visible"])
        mean_area = float(stats["mean_area"])
        max_area = float(stats["max_area"])
        drop_reason: Optional[str] = None

        if visible <= 0:
            drop_reason = "empty_local_track"
        elif label_l == "book":
            drop_reason = "disabled_book_label"
        elif (
            j != dominant_person_idx
            and label_l in {"person", "people"}
            and visible <= 6
        ):
            drop_reason = f"very_short_person_fragment_of_local:{dominant_person_idx}"
        elif (
            j != dominant_person_idx
            and label_l in {"person", "people"}
            and visible <= 24
            and mean_area < 0.035
        ):
            drop_reason = f"short_person_fragment_of_local:{dominant_person_idx}"
        elif (
            j != dominant_person_idx
            and label_l in {"person", "people"}
            and visible <= 40
            and mean_area < 0.020
            and max_area < 0.060
        ):
            drop_reason = f"short_low_area_person_fragment_of_local:{dominant_person_idx}"
        elif (
            j != dominant_person_idx
            and label_l in {"person", "people"}
            and mean_area < 0.012
            and max_area < 0.025
            and visible <= max(36, int(0.75 * T))
        ):
            drop_reason = f"tiny_person_shadow_of_local:{dominant_person_idx}"
        elif source_type == "thing_tracked" and sem_group == SEMANTIC_GROUP_STATIC_THING:
            if visible <= 1:
                drop_reason = "single_frame_static_false_positive"
            elif label_l in {"door", "desk", "table", "window"}:
                drop_reason = "disabled_static_prompt_label"
            elif mean_area < 0.004 and max_area < 0.010:
                drop_reason = "tiny_static_fragment"
            elif (
                label_l in {"door", "window"}
                and mean_area >= 0.14
                and max_area >= 0.22
                and visible <= max(8, int(0.92 * T))
            ):
                drop_reason = "large_background_static_drift"
        elif source_type in {"structure_tracked", "stuff_static"}:
            if visible < max(6, int(0.12 * T)):
                drop_reason = "short_structure_or_stuff"

        if drop_reason is not None:
            if len(pruned) < 100:
                pruned.append({
                    "local_index": int(j),
                    "label": label_l,
                    "source_type": source_type,
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "max_area": float(max_area),
                    "reason": drop_reason,
                })
            continue
        keep.append(j)

    if len(keep) == mo.num_masklets:
        return mo, pruned

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=mo.M_mask.device)
    compact = MaskletOutput(
        M_mask=mo.M_mask.index_select(0, keep_tensor).contiguous(),
        V_mask=mo.V_mask.index_select(0, keep_tensor).contiguous(),
        B_mask=mo.B_mask.index_select(0, keep_tensor).contiguous(),
        Q_mask=mo.Q_mask.index_select(0, keep_tensor).contiguous(),
        L_sem=[mo.L_sem[j] for j in keep],
        G_sem=mo.G_sem.index_select(0, keep_tensor).contiguous(),
        W_sem=mo.W_sem.index_select(0, keep_tensor).contiguous(),
        A_ratio=mo.A_ratio.index_select(0, keep_tensor).contiguous(),
        num_masklets=len(keep),
        num_frames=mo.num_frames,
        frame_height=mo.frame_height,
        frame_width=mo.frame_width,
        source_type=[mo.source_type[j] for j in keep],
        birth_frame=[mo.birth_frame[j] for j in keep],
        seed_global_track_idx=[
            mo.seed_global_track_idx[j] if j < len(mo.seed_global_track_idx) else None
            for j in keep
        ],
        debug=dict(mo.debug),
    )
    compact.debug["efficient_premerge_pruned_local_tracks"] = pruned
    compact.debug["efficient_premerge_raw_num_masklets"] = int(mo.num_masklets)
    return compact, pruned


def merge_chunk_masklet_into_global(
    global_tracks: List[Dict],
    mo: MaskletOutput,
    chunk_idx: int,
    start: int,
    end: int,
    total_frames: int,
    chunk_overlap: int,
) -> Dict[str, object]:
    compact_mo, premerge_pruned = _compact_chunk_output_before_global_merge(mo)
    debug = _mature_merge_chunk_masklet_into_global(
        global_tracks,
        compact_mo,
        chunk_idx=chunk_idx,
        start=start,
        end=end,
        total_frames=total_frames,
        chunk_overlap=chunk_overlap,
    )
    debug["num_local_tracks_raw"] = int(mo.num_masklets)
    debug["premerge_pruned_local_tracks"] = premerge_pruned
    return debug


def _efficient_track_mean_area(track: Dict) -> float:
    vals = list(track.get("area_by_frame", {}).values())
    return float(sum(float(v) for v in vals) / max(len(vals), 1)) if vals else 0.0


def _efficient_track_max_area(track: Dict) -> float:
    vals = list(track.get("area_by_frame", {}).values())
    return float(max(float(v) for v in vals)) if vals else 0.0


def _is_person_track(track: Dict) -> bool:
    return str(track.get("L_sem", "")).strip().lower() in _HUMAN_LABELS


def _canonicalize_efficient_human_labels(tracks: List[Dict]) -> Tuple[List[Dict], int]:
    changed = 0
    for track in tracks:
        if _is_person_track(track) and str(track.get("L_sem", "")).strip().lower() != "person":
            track["L_sem"] = "person"
            changed += 1
    return tracks, changed


def _box_center_norm_torch(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a.tolist()]
    bx1, by1, bx2, by2 = [float(v) for v in box_b.tolist()]
    aw = max(ax2 - ax1 + 1.0, 1.0)
    ah = max(ay2 - ay1 + 1.0, 1.0)
    bw = max(bx2 - bx1 + 1.0, 1.0)
    bh = max(by2 - by1 + 1.0, 1.0)
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    dx = abs(acx - bcx) / max(0.5 * (aw + bw), 1.0)
    dy = abs(acy - bcy) / max(0.5 * (ah + bh), 1.0)
    return float(max(dx, dy))


def _efficient_person_duplicate_ratio(candidate: Dict, reference: Dict) -> Tuple[float, int]:
    cand_frames = sorted(candidate.get("mask_by_frame", {}).keys())
    ref_frames = set(reference.get("mask_by_frame", {}).keys())
    shared = [int(t) for t in cand_frames if int(t) in ref_frames]
    if not shared:
        return 0.0, 0

    duplicate = 0
    checked = 0
    cand_mean_area = max(_efficient_track_mean_area(candidate), 1e-6)
    ref_mean_area = max(_efficient_track_mean_area(reference), 1e-6)
    for t in shared:
        checked += 1
        box_iou = _box_iou_xyxy_torch(
            candidate["box_by_frame"][t],
            reference["box_by_frame"][t],
        )
        center_norm = _box_center_norm_torch(
            candidate["box_by_frame"][t],
            reference["box_by_frame"][t],
        )
        area_sim = min(
            float(candidate["area_by_frame"][t]),
            float(reference["area_by_frame"][t]),
        ) / max(
            float(candidate["area_by_frame"][t]),
            float(reference["area_by_frame"][t]),
            1e-6,
        )
        small_fragment = cand_mean_area < 0.45 * ref_mean_area or cand_mean_area < 0.018
        if (
            box_iou >= 0.32
            or (small_fragment and box_iou >= 0.16 and center_norm <= 0.85)
            or (small_fragment and box_iou >= 0.28 and area_sim >= 0.20)
            or (small_fragment and center_norm <= 0.42 and area_sim >= 0.25)
        ):
            duplicate += 1

    return duplicate / max(checked, 1), checked


def _track_visible_len(track: Dict) -> int:
    return int(len(track.get("mask_by_frame", {})))


def _delete_track_frame(track: Dict, frame_idx: int) -> None:
    frame_idx = int(frame_idx)
    track.get("mask_by_frame", {}).pop(frame_idx, None)
    track.get("box_by_frame", {}).pop(frame_idx, None)
    track.get("q_by_frame", {}).pop(frame_idx, None)
    track.get("area_by_frame", {}).pop(frame_idx, None)


def _packed_mask_alignment(track_a: Dict, track_b: Dict, frame_idx: int) -> Tuple[float, float, float]:
    """Return IoU plus A/B coverage for one packed-mask frame."""
    frame_idx = int(frame_idx)
    if frame_idx not in track_a.get("mask_by_frame", {}) or frame_idx not in track_b.get("mask_by_frame", {}):
        return 0.0, 0.0, 0.0
    H = int(track_a.get("frame_height", track_b.get("frame_height", 0)))
    W = int(track_a.get("frame_width", track_b.get("frame_width", 0)))
    if H <= 0 or W <= 0:
        return 0.0, 0.0, 0.0
    mask_a = _unpack_mask_np(track_a["mask_by_frame"][frame_idx], H, W)
    mask_b = _unpack_mask_np(track_b["mask_by_frame"][frame_idx], H, W)
    inter = float((mask_a & mask_b).sum())
    area_a = float(mask_a.sum())
    area_b = float(mask_b.sum())
    union = area_a + area_b - inter
    iou = inter / union if union > 0.0 else 0.0
    cover_a = inter / area_a if area_a > 0.0 else 0.0
    cover_b = inter / area_b if area_b > 0.0 else 0.0
    return float(iou), float(cover_a), float(cover_b)


def _mask_to_box_tensor(mask: np.ndarray) -> torch.Tensor:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return torch.zeros(4, dtype=torch.float32)
    return torch.tensor(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=torch.float32,
    )


def _subtract_thing_masks_from_structure_tracks(
    tracks: List[Dict],
) -> Tuple[List[Dict], Dict[str, object]]:
    """Remove foreground thing pixels from global structure tracks.

    Cutie tracks floor/wall/ceiling as regular objects, so their masks can
    overlap people in the final sparse output. Keeping the overlap hurts both
    visual inspection and downstream semantic use. This pass only edits
    structure/stuff masks, and keeps a frame if enough structure remains.
    """
    frame_to_thing_masks: Dict[int, List[Tuple[np.ndarray, int, int]]] = {}
    frame_to_structure_tracks: Dict[int, List[Dict]] = {}
    for track in tracks:
        source_type = str(track.get("source_type", ""))
        H = int(track.get("frame_height", 0))
        W = int(track.get("frame_width", 0))
        if H <= 0 or W <= 0:
            continue
        if source_type == "thing_tracked":
            for frame_idx, packed in track.get("mask_by_frame", {}).items():
                frame_to_thing_masks.setdefault(int(frame_idx), []).append((packed, H, W))
        elif source_type in {"structure_tracked", "stuff_static"}:
            for frame_idx in track.get("mask_by_frame", {}).keys():
                frame_to_structure_tracks.setdefault(int(frame_idx), []).append(track)

    shared_frames = sorted(set(frame_to_thing_masks).intersection(frame_to_structure_tracks))
    if not shared_frames:
        return tracks, {"frames_cleaned": 0, "frames_deleted": 0, "pixels_removed": 0}

    frames_cleaned = 0
    frames_deleted = 0
    pixels_removed = 0

    for t in shared_frames:
        structure_tracks = frame_to_structure_tracks.get(t, [])
        if not structure_tracks:
            continue
        Ht = int(structure_tracks[0].get("frame_height", 0))
        Wt = int(structure_tracks[0].get("frame_width", 0))
        if Ht <= 0 or Wt <= 0:
            continue
        thing_union = np.zeros((Ht, Wt), dtype=bool)
        for packed, Hm, Wm in frame_to_thing_masks.get(t, []):
            if Hm != Ht or Wm != Wt:
                continue
            thing_union |= _unpack_mask_np(packed, Hm, Wm)
        if thing_union.sum() <= 0:
            continue

        for track in structure_tracks:
            if t not in track.get("mask_by_frame", {}):
                continue
            mask = _unpack_mask_np(track["mask_by_frame"][t], Ht, Wt)
            original_area = float(mask.sum())
            if original_area <= 0.0:
                _delete_track_frame(track, t)
                frames_deleted += 1
                continue
            cleaned = mask & (~thing_union)
            cleaned_area = float(cleaned.sum())
            removed = int(original_area - cleaned_area)
            if removed <= 0:
                continue
            pixels_removed += removed
            if cleaned_area < max(128.0, 0.12 * original_area):
                _delete_track_frame(track, t)
                frames_deleted += 1
                continue
            track["mask_by_frame"][t] = _pack_mask_np(cleaned)
            track["box_by_frame"][t] = _mask_to_box_tensor(cleaned)
            track["area_by_frame"][t] = float(cleaned_area / max(float(Ht * Wt), 1.0))
            frames_cleaned += 1

    return tracks, {
        "frames_cleaned": int(frames_cleaned),
        "frames_deleted": int(frames_deleted),
        "pixels_removed": int(pixels_removed),
    }


def _keep_largest_structure_components(
    mask: np.ndarray,
    *,
    max_components: int = 3,
    min_component_area: int = 192,
) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    if num_labels <= 1:
        return mask.astype(bool)
    components: List[Tuple[int, int]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area >= int(min_component_area):
            components.append((area, label_idx))
    if not components:
        return np.zeros_like(mask, dtype=bool)
    components.sort(reverse=True)
    keep = np.zeros_like(mask, dtype=bool)
    for _area, label_idx in components[:max_components]:
        keep |= labels == int(label_idx)
    return keep


def _clean_structure_geometry_frames(
    tracks: List[Dict],
) -> Tuple[List[Dict], Dict[str, object]]:
    """Clamp efficient-path STRUCTURE/STUFF masks to conservative geometry.

    The light tracker is fast enough for Taylor, but floor/ceiling masks can
    drift into large semi-transparent sheets. For the efficient path, false
    STUFF is worse than missing STUFF, so apply a strict per-frame geometry
    gate before serializing the sparse output.
    """
    frames_cleaned = 0
    frames_deleted = 0
    pixels_removed = 0
    examples: List[Dict[str, object]] = []

    for track_idx, track in enumerate(tracks):
        source_type = str(track.get("source_type", ""))
        if source_type not in {"structure_tracked", "stuff_static"}:
            continue
        label_l = str(track.get("L_sem", "")).strip().lower()
        if label_l not in {"floor", "ceiling", "wall"}:
            continue

        H = max(int(track.get("frame_height", 0)), 1)
        W = max(int(track.get("frame_width", 0)), 1)
        for frame_idx in list(track.get("mask_by_frame", {}).keys()):
            t = int(frame_idx)
            packed = track.get("mask_by_frame", {}).get(t)
            if packed is None:
                continue
            mask = _unpack_mask_np(packed, H, W).astype(bool)
            original_area = int(mask.sum())
            if original_area <= 0:
                _delete_track_frame(track, t)
                frames_deleted += 1
                continue

            cleaned = mask
            if label_l == "floor":
                allowed = np.zeros((H, W), dtype=bool)
                allowed[int(0.42 * H):, :] = True
                cleaned = cleaned & allowed
            elif label_l == "ceiling":
                allowed = np.zeros((H, W), dtype=bool)
                allowed[: int(0.43 * H), :] = True
                cleaned = cleaned & allowed
            elif label_l == "wall":
                # Wall masks from the efficient path are especially ambiguous:
                # drop obvious floor/ceiling halves but keep vertical surfaces.
                allowed = np.ones((H, W), dtype=bool)
                allowed[: int(0.06 * H), :] = False
                allowed[int(0.88 * H):, :] = False
                cleaned = cleaned & allowed

            cleaned_area = int(cleaned.sum())
            if cleaned_area < max(256, int(0.18 * original_area)):
                _delete_track_frame(track, t)
                frames_deleted += 1
                if len(examples) < 120:
                    examples.append({
                        "track_index": int(track_idx),
                        "frame_idx": int(t),
                        "label": label_l,
                        "reason": "structure_geometry_too_small_after_clip",
                        "original_area": int(original_area),
                        "cleaned_area": int(cleaned_area),
                    })
                continue

            box = _mask_to_box_tensor(cleaned)
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max((x2 - x1 + 1.0) / max(float(W), 1.0), 1e-6)
            bh = max((y2 - y1 + 1.0) / max(float(H), 1.0), 1e-6)
            y_center = (y1 + y2) / max(2.0 * float(H), 1.0)
            area_ratio = float(cleaned_area) / max(float(H * W), 1.0)
            top_touch = y1 <= 3.0
            bottom_touch = y2 >= H - 4.0

            bad_geometry = False
            reason = ""
            if label_l == "floor":
                bad_geometry = (
                    y_center < 0.58
                    or (not bottom_touch and y_center < 0.68 and bh > 0.22)
                    or area_ratio > 0.18
                    or (bh > 0.62 and bw > 0.72)
                )
                reason = "implausible_floor_frame"
            elif label_l == "ceiling":
                bad_geometry = (
                    y_center > 0.30
                    or bottom_touch
                    or area_ratio > 0.14
                    or bh > 0.44
                )
                reason = "implausible_ceiling_frame"
            elif label_l == "wall":
                bad_geometry = (
                    area_ratio > 0.16
                    or (top_touch and bottom_touch)
                    or y_center < 0.18
                    or y_center > 0.82
                )
                reason = "implausible_wall_frame"

            if bad_geometry:
                _delete_track_frame(track, t)
                frames_deleted += 1
                if len(examples) < 120:
                    examples.append({
                        "track_index": int(track_idx),
                        "frame_idx": int(t),
                        "label": label_l,
                        "reason": reason,
                        "area_ratio": float(area_ratio),
                        "box_width": float(bw),
                        "box_height": float(bh),
                        "y_center": float(y_center),
                    })
                continue

            if cleaned_area != original_area:
                pixels_removed += int(original_area - cleaned_area)
                frames_cleaned += 1
                track["mask_by_frame"][t] = _pack_mask_np(cleaned)
                track["box_by_frame"][t] = box
                track["area_by_frame"][t] = float(area_ratio)

    return tracks, {
        "frames_cleaned": int(frames_cleaned),
        "frames_deleted": int(frames_deleted),
        "pixels_removed": int(pixels_removed),
        "examples": examples,
    }


def _copy_track_frame(track: Dict, dst_t: int, src_t: int, *, q_scale: float = 0.98) -> None:
    dst_t = int(dst_t)
    src_t = int(src_t)
    packed = track["mask_by_frame"][src_t]
    track["mask_by_frame"][dst_t] = packed.copy() if hasattr(packed, "copy") else packed
    box = track["box_by_frame"][src_t]
    track["box_by_frame"][dst_t] = box.clone() if hasattr(box, "clone") else torch.as_tensor(box).clone()
    track["q_by_frame"][dst_t] = min(float(track["q_by_frame"].get(src_t, 1.0)) * float(q_scale), 1.0)
    track["area_by_frame"][dst_t] = float(track["area_by_frame"].get(src_t, 0.0))


def _repair_efficient_person_temporal_collapses(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    debug: List[Dict[str, object]] = []
    for idx, track in enumerate(tracks):
        if not _is_person_track(track):
            continue
        frames = sorted(int(t) for t in track.get("mask_by_frame", {}).keys())
        if len(frames) < 3:
            continue
        frame_set = set(frames)

        # Fill tiny gaps only when both neighbours are large and spatially
        # consistent. This fixes 1-frame EfficientSAM3 flicker without turning
        # detector support into long frozen masks.
        for left, right in zip(frames[:-1], frames[1:]):
            gap = int(right) - int(left)
            if gap <= 1 or gap > 3:
                continue
            left_area = float(track.get("area_by_frame", {}).get(left, 0.0))
            right_area = float(track.get("area_by_frame", {}).get(right, 0.0))
            if min(left_area, right_area) < 0.018:
                continue
            if _box_center_norm_torch(track["box_by_frame"][left], track["box_by_frame"][right]) > 0.75:
                continue
            for t in range(left + 1, right):
                src = left if (t - left) <= (right - t) else right
                _copy_track_frame(track, t, src, q_scale=0.96)
                if len(debug) < 200:
                    debug.append({
                        "track_index": int(idx),
                        "frame_idx": int(t),
                        "source_frame": int(src),
                        "reason": "fill_short_gap",
                    })

        frames = sorted(int(t) for t in track.get("mask_by_frame", {}).keys())
        frame_set = set(frames)
        for t in frames:
            area = float(track.get("area_by_frame", {}).get(t, 0.0))
            if area <= 0.0:
                continue
            prev_candidates = [p for p in (t - 1, t - 2) if p in frame_set]
            next_candidates = [n for n in (t + 1, t + 2) if n in frame_set]
            prev_t = max(prev_candidates) if prev_candidates else None
            next_t = min(next_candidates) if next_candidates else None
            if prev_t is None or next_t is None:
                continue
            prev_area = float(track.get("area_by_frame", {}).get(prev_t, 0.0))
            next_area = float(track.get("area_by_frame", {}).get(next_t, 0.0))
            neighbour_area = max(prev_area, next_area)
            if neighbour_area < max(area * 2.8, 0.018):
                continue
            if _box_center_norm_torch(track["box_by_frame"][prev_t], track["box_by_frame"][next_t]) > 0.75:
                continue
            src = prev_t if prev_area >= next_area else next_t
            _copy_track_frame(track, t, src, q_scale=0.97)
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "frame_idx": int(t),
                    "source_frame": int(src),
                    "old_area": float(area),
                    "new_area": float(track.get("area_by_frame", {}).get(t, 0.0)),
                    "reason": "replace_short_collapse",
                })

    return tracks, debug


def _warp_sparse_mask_between_boxes(
    mask: np.ndarray,
    src_box: torch.Tensor,
    dst_box: torch.Tensor,
    H: int,
    W: int,
) -> np.ndarray:
    src = src_box.detach().cpu().numpy().astype(np.float32)
    dst = dst_box.detach().cpu().numpy().astype(np.float32)
    src_w = max(float(src[2] - src[0] + 1.0), 1.0)
    src_h = max(float(src[3] - src[1] + 1.0), 1.0)
    dst_w = max(float(dst[2] - dst[0] + 1.0), 1.0)
    dst_h = max(float(dst[3] - dst[1] + 1.0), 1.0)
    sx = float(np.clip(dst_w / src_w, 0.60, 1.65))
    sy = float(np.clip(dst_h / src_h, 0.60, 1.65))
    src_cx = 0.5 * float(src[0] + src[2])
    src_cy = 0.5 * float(src[1] + src[3])
    dst_cx = 0.5 * float(dst[0] + dst[2])
    dst_cy = 0.5 * float(dst[1] + dst[3])
    matrix = np.asarray(
        [
            [sx, 0.0, dst_cx - sx * src_cx],
            [0.0, sy, dst_cy - sy * src_cy],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(
        np.asarray(mask).astype(np.uint8),
        matrix,
        (int(W), int(H)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (warped > 0).astype(bool)


def _repair_efficient_person_long_gaps(
    tracks: List[Dict],
    *,
    total_frames: int,
    max_gap: int = 36,
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    debug: List[Dict[str, object]] = []
    for idx, track in enumerate(tracks):
        if not _is_person_track(track):
            continue
        frames = sorted(int(t) for t in track.get("mask_by_frame", {}).keys())
        if len(frames) < 2:
            continue
        H = int(track.get("frame_height", 0))
        W = int(track.get("frame_width", 0))
        if H <= 0 or W <= 0:
            continue
        for left, right in zip(frames[:-1], frames[1:]):
            gap = int(right) - int(left)
            if gap <= 1 or gap > int(max_gap):
                continue
            if left not in track.get("box_by_frame", {}) or right not in track.get("box_by_frame", {}):
                continue
            left_area = float(track.get("area_by_frame", {}).get(left, 0.0))
            right_area = float(track.get("area_by_frame", {}).get(right, 0.0))
            if min(left_area, right_area) < 0.018:
                continue
            area_sim = min(left_area, right_area) / max(left_area, right_area, 1e-6)
            center_norm = _box_center_norm_torch(track["box_by_frame"][left], track["box_by_frame"][right])
            if area_sim < 0.24 or center_norm > 0.90:
                continue
            left_mask = _unpack_mask_np(track["mask_by_frame"][left], H, W)
            right_mask = _unpack_mask_np(track["mask_by_frame"][right], H, W)
            if left_mask.sum() <= 0 or right_mask.sum() <= 0:
                continue
            for t in range(left + 1, right):
                if t < 0 or t >= int(total_frames) or t in track["mask_by_frame"]:
                    continue
                alpha = float(t - left) / float(gap)
                target_box = (1.0 - alpha) * track["box_by_frame"][left] + alpha * track["box_by_frame"][right]
                if alpha <= 0.5:
                    src_mask = left_mask
                    src_box = track["box_by_frame"][left]
                    src_q = float(track["q_by_frame"].get(left, 1.0))
                else:
                    src_mask = right_mask
                    src_box = track["box_by_frame"][right]
                    src_q = float(track["q_by_frame"].get(right, 1.0))
                warped = _warp_sparse_mask_between_boxes(src_mask, src_box, target_box, H, W)
                if warped.sum() <= 0:
                    continue
                track["mask_by_frame"][int(t)] = _pack_mask_np(warped)
                track["box_by_frame"][int(t)] = _mask_to_box_tensor(warped)
                track["q_by_frame"][int(t)] = min(src_q * 0.94, 1.0)
                track["area_by_frame"][int(t)] = float(warped.sum()) / max(float(H * W), 1.0)
                if len(debug) < 200:
                    debug.append({
                        "track_index": int(idx),
                        "frame_idx": int(t),
                        "left": int(left),
                        "right": int(right),
                        "gap": int(gap),
                        "area_sim": float(area_sim),
                        "center_norm": float(center_norm),
                    })
    return tracks, debug


def _suppress_efficient_duplicate_person_frames(
    tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    debug: List[Dict[str, object]] = []
    suppressed = 0

    def _frame_person_quality(idx: int, t: int) -> float:
        track = tracks[idx]
        area = float(track.get("area_by_frame", {}).get(t, 0.0))
        q = float(track.get("q_by_frame", {}).get(t, 0.0))
        visible_bonus = min(float(_track_visible_len(track)) / max(float(total_frames), 1.0), 1.0)
        mean_area = _efficient_track_mean_area(track)
        box = track.get("box_by_frame", {}).get(t)
        if box is None:
            return q + 0.10 * visible_bonus

        box_vals = box.detach().cpu().numpy() if hasattr(box, "detach") else np.asarray(box)
        x1, y1, x2, y2 = [float(v) for v in box_vals[:4]]
        W = max(float(track.get("frame_width", 0)), 1.0)
        H = max(float(track.get("frame_height", 0)), 1.0)
        bw = max((x2 - x1 + 1.0) / W, 1e-6)
        bh = max((y2 - y1 + 1.0) / H, 1e-6)
        aspect = bh / max(bw, 1e-6)

        # Person-sized masks are useful; huge wide masks are often tracker drift
        # into curtain/floor/background, so do not let "largest area wins" decide.
        size_score = min(area / 0.080, 1.0)
        if area > 0.26:
            size_score -= min((area - 0.26) / 0.20, 1.0) * 0.50
        aspect_score = 1.0 - min(abs(aspect - 2.0) / 2.2, 1.0)
        wide_drift_penalty = 0.0
        if area > 0.24 and bw > 0.42 and aspect < 1.35:
            wide_drift_penalty += 0.45
        if area > 0.34 and (x1 <= 2.0 or x2 >= W - 3.0):
            wide_drift_penalty += 0.35

        return (
            0.48 * q
            + 0.22 * size_score
            + 0.14 * aspect_score
            + 0.10 * visible_bonus
            + 0.06 * min(mean_area / 0.10, 1.0)
            - wide_drift_penalty
        )

    frame_to_people: Dict[int, List[int]] = {}
    for idx, track in enumerate(tracks):
        if not _is_person_track(track):
            continue
        for t in track.get("mask_by_frame", {}).keys():
            frame_to_people.setdefault(int(t), []).append(idx)

    for t, indices in frame_to_people.items():
        if len(indices) <= 1:
            continue
        ordered = sorted(
            indices,
            key=lambda idx: (
                -_frame_person_quality(idx, int(t)),
                -float(tracks[idx].get("q_by_frame", {}).get(t, 0.0)),
                -_track_visible_len(tracks[idx]),
                int(tracks[idx].get("birth_frame", 0)),
            ),
        )
        kept: List[int] = []
        for idx in ordered:
            track = tracks[idx]
            if t not in track.get("box_by_frame", {}):
                continue
            cand_box = track["box_by_frame"][t]
            cand_area = float(track.get("area_by_frame", {}).get(t, 0.0))
            cand_mean = _efficient_track_mean_area(track)
            duplicate_against: Optional[int] = None
            duplicate_stats: Dict[str, float] = {}
            for kept_idx in kept:
                kept_track = tracks[kept_idx]
                if t not in kept_track.get("box_by_frame", {}):
                    continue
                kept_box = kept_track["box_by_frame"][t]
                kept_area = float(kept_track.get("area_by_frame", {}).get(t, 0.0))
                box_iou = _box_iou_xyxy_torch(cand_box, kept_box)
                center_norm = _box_center_norm_torch(cand_box, kept_box)
                area_sim = min(cand_area, kept_area) / max(cand_area, kept_area, 1e-6)
                mask_iou, cand_cover, kept_cover = _packed_mask_alignment(track, kept_track, int(t))
                small_fragment = (
                    cand_mean < 0.55 * max(_efficient_track_mean_area(kept_track), 1e-6)
                    or cand_area < 0.018
                    or cand_area < 0.72 * max(kept_area, 1e-6)
                )
                duplicate = (
                    box_iou >= 0.35
                    or (box_iou >= 0.18 and area_sim >= 0.28 and center_norm <= 0.80)
                    or (box_iou >= 0.10 and mask_iou >= 0.08 and center_norm <= 0.70)
                    or (center_norm <= 0.52 and mask_iou >= 0.06 and area_sim >= 0.20)
                    or (center_norm <= 0.78 and cand_cover >= 0.34 and area_sim >= 0.18)
                    or (small_fragment and center_norm <= 0.55 and area_sim >= 0.20)
                    or (small_fragment and box_iou >= 0.10 and center_norm <= 0.95)
                    or (small_fragment and cand_cover >= 0.55)
                    or (small_fragment and mask_iou >= 0.14 and center_norm <= 0.95)
                )
                if duplicate:
                    duplicate_against = kept_idx
                    duplicate_stats = {
                        "box_iou": float(box_iou),
                        "center_norm": float(center_norm),
                        "area_sim": float(area_sim),
                        "mask_iou": float(mask_iou),
                        "cand_cover": float(cand_cover),
                        "kept_cover": float(kept_cover),
                    }
                    break
            if duplicate_against is not None:
                _delete_track_frame(track, t)
                suppressed += 1
                if len(debug) < 200:
                    debug.append({
                        "frame_idx": int(t),
                        "track_index": int(idx),
                        "against": int(duplicate_against),
                        **duplicate_stats,
                    })
                continue
            kept.append(idx)

    debug.append({"num_frames_suppressed": int(suppressed)})
    return tracks, debug


def _prune_wide_person_drift_frames(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    debug: List[Dict[str, object]] = []
    removed_frames = 0
    for idx, track in enumerate(tracks):
        if not _is_person_track(track):
            continue
        track_areas = [
            float(v)
            for v in track.get("area_by_frame", {}).values()
            if float(v) > 0.0
        ]
        median_area = float(np.median(track_areas)) if track_areas else 0.0
        H = max(int(track.get("frame_height", 0)), 1)
        W = max(int(track.get("frame_width", 0)), 1)
        for t in list(track.get("box_by_frame", {}).keys()):
            t = int(t)
            box = track["box_by_frame"].get(t)
            if box is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max((x2 - x1 + 1.0) / max(float(W), 1.0), 1e-6)
            bh = max((y2 - y1 + 1.0) / max(float(H), 1.0), 1e-6)
            aspect = bh / max(bw, 1e-6)
            area = float(track.get("area_by_frame", {}).get(t, 0.0))
            touches_both_sides = x1 <= 3.0 and x2 >= W - 4.0
            touches_edge = x1 <= 3.0 or y1 <= 3.0 or x2 >= W - 4.0 or y2 >= H - 4.0

            wide_background_drift = (
                area >= 0.30
                and bw >= 0.72
                and aspect <= 0.90
            )
            full_width_drift = (
                area >= 0.24
                and touches_both_sides
                and aspect <= 1.15
            )
            tall_broad_drift = (
                area >= 0.24
                and bw >= 0.58
                and bh >= 0.86
                and touches_edge
            )
            sudden_area_spike = (
                median_area > 0.0
                and area >= max(0.22, 2.15 * median_area)
                and bw >= 0.52
                and bh >= 0.68
                and touches_edge
            )
            oversized_person_sheet = (
                area >= 0.34
                and bw >= 0.45
                and bh >= 0.82
            )
            if not (
                wide_background_drift
                or full_width_drift
                or tall_broad_drift
                or sudden_area_spike
                or oversized_person_sheet
            ):
                continue

            _delete_track_frame(track, t)
            removed_frames += 1
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "frame_idx": int(t),
                    "area": float(area),
                    "box_width": float(bw),
                    "box_height": float(bh),
                    "aspect": float(aspect),
                    "median_area": float(median_area),
                    "reason": "wide_person_background_drift",
                })

    debug.append({"num_frames_removed": int(removed_frames)})
    return tracks, debug


def _prune_weak_efficient_structure_tracks(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    kept: List[Dict] = []
    debug: List[Dict[str, object]] = []
    for idx, track in enumerate(tracks):
        source_type = str(track.get("source_type", ""))
        if source_type not in {"stuff_static", "structure_tracked"}:
            kept.append(track)
            continue
        visible = _track_visible_len(track)
        mean_q = (
            sum(float(v) for v in track.get("q_by_frame", {}).values()) / max(visible, 1)
            if visible > 0 else 0.0
        )
        mean_area = _efficient_track_mean_area(track)
        drop = visible <= 3 or (visible <= 5 and mean_q < 0.36 and mean_area > 0.08)
        if source_type == "stuff_static":
            # The efficient path only has sparse YOLOE support for stuff. Low
            # confidence stuff masks are more harmful than useful because they
            # show up as large, sticky overlays in the rendered video.
            label_l = str(track.get("L_sem", "")).strip().lower()
            is_structure_group = int(track.get("G_sem", -1)) == 0
            structure_area_cap = {
                "floor": 0.20,
                "wall": 0.18,
                "ceiling": 0.16,
            }.get(label_l, 0.30)
            reliable_flow_structure = (
                is_structure_group
                and visible >= 6
                and mean_q >= 0.16
                and 0.006 <= mean_area <= structure_area_cap
            )
            geometry_invalid_structure = False
            if is_structure_group and label_l in {"wall", "floor", "ceiling"}:
                # The YOLOE/static efficient STUFF path is still too coarse for
                # Taylor: it produces large, low-confidence sheets that look
                # like semantic segmentation but are often wrong. Keep the
                # tracked SAM3/EdgeTAM structure path, but reject these sparse
                # static floor/wall/ceiling masks unless they become genuinely
                # high confidence in another model/config.
                if mean_q < 0.35:
                    reliable_flow_structure = False
                    drop = True
                # Large structure masks are the main failure mode of the
                # efficient path: once accepted, optical-flow STUFF propagation
                # makes them sticky for hundreds of frames. Prefer missing a
                # coarse wall over flooding the output with a wrong wall.
                if mean_area > structure_area_cap:
                    drop = True
                stats = _structure_geometry_stats(track)
                if stats:
                    mean_y = float(stats.get("mean_y_center", 0.0))
                    mean_height = float(stats.get("mean_height", 0.0))
                    max_area = float(stats.get("max_area", 0.0))
                    bottom_touch = float(stats.get("bottom_touch_ratio", 0.0))
                    if label_l == "ceiling" and (
                        mean_height > 0.50
                        or mean_y > 0.28
                        or bottom_touch > 0.05
                    ):
                        geometry_invalid_structure = True
                    if label_l == "floor" and (
                        mean_y < 0.58
                        or (mean_height > 0.70 and max_area > 0.16)
                    ):
                        geometry_invalid_structure = True
                if geometry_invalid_structure:
                    reliable_flow_structure = False
                    drop = True
            # Curtain/screen-style low-value STUFF is too scene-specific for a
            # default front-end. On Taylor it often fires on hallway walls and
            # then pollutes many frames, so keep the default efficient path
            # focused on geometric STRUCTURE labels.
            reliable_curtain = False
            reliable_sparse_stuff = (
                visible >= 3
                and mean_q >= 0.70
                and 0.008 <= mean_area <= 0.12
            )
            if reliable_flow_structure or reliable_curtain:
                drop = False
            else:
                drop = (
                    not reliable_sparse_stuff
                    and (drop or mean_q < 0.45 or (visible <= 8 and mean_area > 0.12))
                )
        if drop:
            if len(debug) < 100:
                debug.append({
                    "track_index": int(idx),
                    "label": str(track.get("L_sem", "")),
                    "source_type": source_type,
                    "visible_frames": int(visible),
                    "mean_q": float(mean_q),
                    "mean_area": float(mean_area),
                })
            continue
        kept.append(track)
    return kept, debug


def _prune_excess_efficient_stuff_tracks(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    """Keep curtain/stuff recall without flooding the video with fragments."""
    grouped: Dict[str, List[int]] = {}
    for idx, track in enumerate(tracks):
        if str(track.get("source_type", "")) != "stuff_static":
            continue
        label_l = str(track.get("L_sem", "")).strip().lower()
        if label_l in {"curtain"}:
            grouped.setdefault(label_l, []).append(idx)

    removed: set[int] = set()
    debug: List[Dict[str, object]] = []
    max_by_label = {"curtain": 8}

    for label_l, indices in grouped.items():
        scored: List[Tuple[float, int]] = []
        for idx in indices:
            track = tracks[idx]
            visible = _track_visible_len(track)
            mean_area = _efficient_track_mean_area(track)
            mean_q = (
                sum(float(v) for v in track.get("q_by_frame", {}).values()) / max(visible, 1)
                if visible > 0 else 0.0
            )
            if visible < 48 and mean_area < 0.060:
                removed.add(idx)
                reason = "short_curtain_fragment"
            elif mean_area < 0.012 and visible < 180:
                removed.add(idx)
                reason = "tiny_curtain_fragment"
            else:
                score = float(visible) * max(float(mean_area), 1e-4) ** 0.5 * max(float(mean_q), 0.05)
                scored.append((score, idx))
                continue
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": label_l,
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "mean_q": float(mean_q),
                    "reason": reason,
                })

        scored.sort(reverse=True)
        keep_set = {idx for _, idx in scored[:max_by_label.get(label_l, len(scored))]}
        for _, idx in scored[max_by_label.get(label_l, len(scored)):]:
            removed.add(idx)
            track = tracks[idx]
            visible = _track_visible_len(track)
            mean_area = _efficient_track_mean_area(track)
            mean_q = (
                sum(float(v) for v in track.get("q_by_frame", {}).values()) / max(visible, 1)
                if visible > 0 else 0.0
            )
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": label_l,
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "mean_q": float(mean_q),
                    "reason": "excess_curtain_fragment",
                })

    kept = [track for idx, track in enumerate(tracks) if idx not in removed]
    return kept, debug


def _structure_geometry_stats(track: Dict) -> Dict[str, float]:
    frames = sorted(int(t) for t in track.get("box_by_frame", {}).keys())
    if not frames:
        return {}
    H = max(int(track.get("frame_height", 0)), 1)
    W = max(int(track.get("frame_width", 0)), 1)
    sample = frames
    if len(sample) > 48:
        positions = np.linspace(0, len(sample) - 1, 48)
        sample = [sample[int(round(pos))] for pos in positions]

    y_centers: List[float] = []
    heights: List[float] = []
    widths: List[float] = []
    top_touch = 0
    bottom_touch = 0
    left_touch = 0
    right_touch = 0
    for t in sample:
        box = track["box_by_frame"].get(t)
        if box is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        y_centers.append((y1 + y2) / max(2.0 * H, 1.0))
        heights.append((y2 - y1 + 1.0) / max(float(H), 1.0))
        widths.append((x2 - x1 + 1.0) / max(float(W), 1.0))
        top_touch += int(y1 <= 3.0)
        bottom_touch += int(y2 >= H - 4.0)
        left_touch += int(x1 <= 3.0)
        right_touch += int(x2 >= W - 4.0)

    n = max(len(y_centers), 1)
    areas = [float(v) for v in track.get("area_by_frame", {}).values()]
    return {
        "visible_frames": float(len(frames)),
        "mean_area": float(sum(areas) / max(len(areas), 1)) if areas else 0.0,
        "max_area": float(max(areas)) if areas else 0.0,
        "mean_y_center": float(sum(y_centers) / n),
        "mean_height": float(sum(heights) / max(len(heights), 1)) if heights else 0.0,
        "mean_width": float(sum(widths) / max(len(widths), 1)) if widths else 0.0,
        "top_touch_ratio": float(top_touch / n),
        "bottom_touch_ratio": float(bottom_touch / n),
        "left_touch_ratio": float(left_touch / n),
        "right_touch_ratio": float(right_touch / n),
    }


def _regularize_efficient_structure_semantics(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    kept: List[Dict] = []
    debug: List[Dict[str, object]] = []
    structure_labels = {"floor", "wall", "ceiling"}

    for idx, track in enumerate(tracks):
        source_type = str(track.get("source_type", ""))
        label_l = str(track.get("L_sem", "")).strip().lower()
        if source_type not in {"structure_tracked", "stuff_static"} or label_l not in structure_labels:
            kept.append(track)
            continue

        stats = _structure_geometry_stats(track)
        if not stats:
            continue

        mean_y = float(stats["mean_y_center"])
        mean_height = float(stats["mean_height"])
        mean_width = float(stats["mean_width"])
        mean_area = float(stats["mean_area"])
        max_area = float(stats["max_area"])
        top_touch = float(stats["top_touch_ratio"])
        bottom_touch = float(stats["bottom_touch_ratio"])

        drop_reason: Optional[str] = None
        new_label = label_l

        full_frame_wash = (
            (max_area >= 0.58 and top_touch >= 0.55 and bottom_touch >= 0.45)
            or (mean_height >= 0.92 and mean_width >= 0.62 and top_touch >= 0.70 and bottom_touch >= 0.70)
        )
        if full_frame_wash:
            drop_reason = "ambiguous_full_frame_structure"
        elif bottom_touch >= 0.45 and mean_y >= 0.58 and top_touch < 0.35:
            new_label = "floor"
        elif top_touch >= 0.35 and mean_y <= 0.32 and bottom_touch < 0.20:
            new_label = "ceiling"
        elif label_l == "floor" and mean_y < 0.50 and bottom_touch < 0.30:
            new_label = "wall"
        elif label_l == "ceiling" and (mean_y > 0.42 or bottom_touch > 0.22):
            new_label = "wall" if mean_y < 0.62 else "floor"
        elif label_l == "wall" and bottom_touch >= 0.60 and mean_y >= 0.66:
            new_label = "floor"
        elif label_l == "wall" and top_touch >= 0.65 and mean_y <= 0.24 and mean_area < 0.16:
            new_label = "ceiling"

        if drop_reason is None:
            if new_label == "ceiling" and (
                mean_height > 0.50
                or mean_y > 0.28
                or bottom_touch > 0.05
            ):
                drop_reason = "implausible_ceiling_geometry"
            elif new_label == "floor" and (
                mean_y < 0.58
                or (mean_height > 0.70 and max_area > 0.16)
            ):
                drop_reason = "implausible_floor_geometry"

        if drop_reason is not None:
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": label_l,
                    "action": "drop",
                    "reason": drop_reason,
                    **stats,
                })
            continue

        if new_label != label_l:
            track = dict(track)
            track["L_sem"] = new_label
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": label_l,
                    "new_label": new_label,
                    "action": "relabel",
                    **stats,
                })
        kept.append(track)

    return kept, debug


def _is_structure_like_track(track: Dict) -> bool:
    label_l = str(track.get("L_sem", "")).strip().lower()
    source_type = str(track.get("source_type", ""))
    return (
        source_type in {"stuff_static", "structure_tracked"}
        and int(track.get("G_sem", -1)) == 0
    )


def _structure_track_boundary_score(left: Dict, right: Dict) -> Tuple[float, Dict[str, float]]:
    if not (_is_structure_like_track(left) and _is_structure_like_track(right)):
        return -1.0, {}
    if not _labels_compatible(str(left.get("L_sem", "")), str(right.get("L_sem", ""))):
        return -1.0, {}

    left_frames = sorted(int(t) for t in left.get("box_by_frame", {}).keys())
    right_frames = sorted(int(t) for t in right.get("box_by_frame", {}).keys())
    if not left_frames or not right_frames:
        return -1.0, {}

    shared = sorted(set(left_frames) & set(right_frames))
    eval_pairs: List[Tuple[int, int]] = [(t, t) for t in shared[:8]]
    if not eval_pairs:
        gap = int(right_frames[0]) - int(left_frames[-1])
        if gap < -6 or gap > 32:
            return -1.0, {}
        for lt in left_frames[-4:]:
            for rt in right_frames[:4]:
                pair_gap = int(rt) - int(lt)
                if -6 <= pair_gap <= 32:
                    eval_pairs.append((int(lt), int(rt)))
    else:
        gap = min(abs(int(right_frames[0]) - int(left_frames[-1])), 0)

    if not eval_pairs:
        return -1.0, {}

    best: Optional[Dict[str, float]] = None
    for lt, rt in eval_pairs:
        if lt not in left.get("box_by_frame", {}) or rt not in right.get("box_by_frame", {}):
            continue
        box_iou = _box_iou_xyxy_torch(left["box_by_frame"][lt], right["box_by_frame"][rt])
        center_norm = _box_center_norm_torch(left["box_by_frame"][lt], right["box_by_frame"][rt])
        left_area = float(left.get("area_by_frame", {}).get(lt, 0.0))
        right_area = float(right.get("area_by_frame", {}).get(rt, 0.0))
        if left_area <= 0.0 or right_area <= 0.0:
            continue
        area_sim = min(left_area, right_area) / max(left_area, right_area, 1e-6)
        score = (
            0.35 * box_iou
            + 0.35 * area_sim
            + 0.25 * max(0.0, 1.0 - min(center_norm, 1.8) / 1.8)
            + 0.05 * max(0.0, 1.0 - max(float(gap), 0.0) / 32.0)
        )
        stats = {
            "left_frame": float(lt),
            "right_frame": float(rt),
            "gap": float(gap),
            "box_iou": float(box_iou),
            "center_norm": float(center_norm),
            "area_sim": float(area_sim),
            "score": float(score),
        }
        if best is None or score > best["score"]:
            best = stats

    if best is None:
        return -1.0, {}
    return float(best["score"]), best


def _merge_adjacent_efficient_structure_tracks(tracks: List[Dict]) -> Tuple[List[Dict], List[Dict[str, object]]]:
    tracks = list(tracks)
    removed: set[int] = set()
    debug: List[Dict[str, object]] = []

    while True:
        best: Optional[Tuple[float, int, int, Dict[str, float]]] = None
        for i, left in enumerate(tracks):
            if i in removed or not _is_structure_like_track(left):
                continue
            for j, right in enumerate(tracks):
                if i == j or j in removed or not _is_structure_like_track(right):
                    continue
                left_frames = sorted(int(t) for t in left.get("box_by_frame", {}).keys())
                right_frames = sorted(int(t) for t in right.get("box_by_frame", {}).keys())
                if not left_frames or not right_frames or left_frames[0] > right_frames[0]:
                    continue
                score, stats = _structure_track_boundary_score(left, right)
                if score < 0.43:
                    continue
                if stats.get("area_sim", 0.0) < 0.24 and stats.get("box_iou", 0.0) < 0.06:
                    continue
                if best is None or score > best[0]:
                    best = (score, i, j, stats)

        if best is None:
            break

        score, primary, secondary, stats = best
        if primary in removed or secondary in removed:
            continue
        if tracks[secondary].get("source_type") == "structure_tracked":
            tracks[primary]["source_type"] = "structure_tracked"
        _merge_track_payload_inplace(tracks[primary], tracks[secondary])
        removed.add(secondary)
        if len(debug) < 200:
            debug.append({
                "primary": int(primary),
                "secondary": int(secondary),
                **stats,
            })

    return [track for idx, track in enumerate(tracks) if idx not in removed], debug


def _person_border_touch_ratio(track: Dict) -> float:
    frames = list(track.get("box_by_frame", {}).keys())
    if not frames:
        return 0.0
    W = max(int(track.get("frame_width", 0)), 1)
    H = max(int(track.get("frame_height", 0)), 1)
    touch = 0
    for t in frames:
        box = track["box_by_frame"][t]
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        if x1 <= 2.0 or y1 <= 2.0 or x2 >= W - 3.0 or y2 >= H - 3.0:
            touch += 1
    return float(touch / max(len(frames), 1))


def _person_box_shape_stats(track: Dict) -> Dict[str, float]:
    frames = list(track.get("box_by_frame", {}).keys())
    if not frames:
        return {
            "mean_aspect": 0.0,
            "wide_ratio": 0.0,
            "mean_bottom": 0.0,
            "mean_center_y": 0.0,
        }

    aspects: List[float] = []
    bottoms: List[float] = []
    center_ys: List[float] = []
    wide = 0
    H = max(int(track.get("frame_height", 0)), 1)
    for t in frames:
        box = track["box_by_frame"][t]
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        bw = max(x2 - x1 + 1.0, 1.0)
        bh = max(y2 - y1 + 1.0, 1.0)
        aspect = bh / bw
        aspects.append(float(aspect))
        bottoms.append(float(y2 / max(float(H - 1), 1.0)))
        center_ys.append(float(((y1 + y2) * 0.5) / max(float(H - 1), 1.0)))
        if aspect < 1.05:
            wide += 1
    return {
        "mean_aspect": float(sum(aspects) / max(len(aspects), 1)),
        "wide_ratio": float(wide / max(len(aspects), 1)),
        "mean_bottom": float(sum(bottoms) / max(len(bottoms), 1)),
        "mean_center_y": float(sum(center_ys) / max(len(center_ys), 1)),
    }


def _prune_edge_stuck_efficient_person_tracks(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    kept: List[Dict] = []
    debug: List[Dict[str, object]] = []
    for idx, track in enumerate(tracks):
        if not _is_person_track(track):
            kept.append(track)
            continue
        visible = _track_visible_len(track)
        mean_area = _efficient_track_mean_area(track)
        max_area = _efficient_track_max_area(track)
        border_ratio = _person_border_touch_ratio(track)
        shape_stats = _person_box_shape_stats(track)
        drop = (
            8 <= visible <= 45
            and border_ratio >= 0.55
            and mean_area < 0.018
            and max_area < 0.055
        )
        if not drop:
            drop = (
                visible <= 32
                and mean_area < 0.018
                and max_area < 0.055
                and border_ratio >= 0.30
                and (
                    float(shape_stats["mean_aspect"]) < 1.35
                    or float(shape_stats["wide_ratio"]) >= 0.20
                )
            )
        if not drop:
            drop = (
                visible <= 10
                and border_ratio >= 0.35
                and mean_area < 0.045
                and max_area < 0.090
            )
        if drop:
            if len(debug) < 100:
                debug.append({
                    "track_index": int(idx),
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "max_area": float(max_area),
                    "border_touch_ratio": float(border_ratio),
                    "mean_aspect": float(shape_stats["mean_aspect"]),
                    "wide_ratio": float(shape_stats["wide_ratio"]),
                    "mean_bottom": float(shape_stats["mean_bottom"]),
                    "mean_center_y": float(shape_stats["mean_center_y"]),
                    "reason": "edge_stuck_person_fragment",
                })
            continue
        kept.append(track)
    return kept, debug


def _merge_track_payload_inplace(primary: Dict, secondary: Dict) -> None:
    primary["birth_frame"] = min(int(primary["birth_frame"]), int(secondary["birth_frame"]))
    primary["W_sem"] = max(float(primary["W_sem"]), float(secondary["W_sem"]))
    for t, packed in secondary.get("mask_by_frame", {}).items():
        t = int(t)
        if t not in primary["mask_by_frame"] or _should_replace_track_frame(primary, secondary, t):
            primary["mask_by_frame"][t] = packed
            primary["box_by_frame"][t] = secondary["box_by_frame"][t]
            primary["q_by_frame"][t] = secondary["q_by_frame"][t]
            primary["area_by_frame"][t] = secondary["area_by_frame"][t]


def _person_frame_duplicate_stats(track_a: Dict, track_b: Dict, frame_idx: int) -> Dict[str, float]:
    box_iou = _box_iou_xyxy_torch(
        track_a["box_by_frame"][frame_idx],
        track_b["box_by_frame"][frame_idx],
    )
    center_norm = _box_center_norm_torch(
        track_a["box_by_frame"][frame_idx],
        track_b["box_by_frame"][frame_idx],
    )
    area_a = float(track_a.get("area_by_frame", {}).get(frame_idx, 0.0))
    area_b = float(track_b.get("area_by_frame", {}).get(frame_idx, 0.0))
    area_sim = min(area_a, area_b) / max(area_a, area_b, 1e-6)
    duplicate = (
        box_iou >= 0.32
        or (box_iou >= 0.18 and center_norm <= 0.78 and area_sim >= 0.25)
        or (center_norm <= 0.42 and area_sim >= 0.22)
    )
    return {
        "box_iou": float(box_iou),
        "center_norm": float(center_norm),
        "area_sim": float(area_sim),
        "duplicate": float(duplicate),
    }


def _person_overlap_merge_score(track_a: Dict, track_b: Dict) -> Tuple[float, Dict[str, float]]:
    shared = sorted(
        set(int(t) for t in track_a.get("mask_by_frame", {}).keys())
        & set(int(t) for t in track_b.get("mask_by_frame", {}).keys())
    )
    if len(shared) < 2:
        return -1.0, {}

    # Sampling keeps the post-process cheap on full videos while still covering
    # long overlaps after repeated chunk stitching.
    if len(shared) > 48:
        sample_pos = np.linspace(0, len(shared) - 1, 48)
        shared_eval = [shared[int(round(pos))] for pos in sample_pos]
    else:
        shared_eval = shared

    dup = 0
    box_ious: List[float] = []
    center_norms: List[float] = []
    area_sims: List[float] = []
    for t in shared_eval:
        if t not in track_a.get("box_by_frame", {}) or t not in track_b.get("box_by_frame", {}):
            continue
        stats = _person_frame_duplicate_stats(track_a, track_b, int(t))
        dup += int(bool(stats["duplicate"]))
        box_ious.append(float(stats["box_iou"]))
        center_norms.append(float(stats["center_norm"]))
        area_sims.append(float(stats["area_sim"]))

    checked = len(box_ious)
    if checked < 2:
        return -1.0, {}

    dup_ratio = dup / max(checked, 1)
    overlap_fraction = len(shared) / max(
        min(_track_visible_len(track_a), _track_visible_len(track_b)),
        1,
    )
    mean_box_iou = sum(box_ious) / checked
    mean_center = sum(center_norms) / checked
    mean_area_sim = sum(area_sims) / checked
    score = (
        0.52 * dup_ratio
        + 0.18 * min(overlap_fraction, 1.0)
        + 0.15 * mean_box_iou
        + 0.10 * mean_area_sim
        + 0.05 * max(0.0, 1.0 - min(mean_center, 1.4) / 1.4)
    )
    return float(score), {
        "shared_frames": float(len(shared)),
        "checked_frames": float(checked),
        "duplicate_ratio": float(dup_ratio),
        "overlap_fraction": float(overlap_fraction),
        "mean_box_iou": float(mean_box_iou),
        "mean_center_norm": float(mean_center),
        "mean_area_sim": float(mean_area_sim),
        "score": float(score),
    }


def _merge_overlapping_efficient_person_tracks(
    tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    tracks = list(tracks)
    removed: set[int] = set()
    debug: List[Dict[str, object]] = []

    while True:
        best: Optional[Tuple[float, int, int, Dict[str, float]]] = None
        for i, track_i in enumerate(tracks):
            if i in removed or not _is_person_track(track_i):
                continue
            for j in range(i + 1, len(tracks)):
                if j in removed or not _is_person_track(tracks[j]):
                    continue
                score, stats = _person_overlap_merge_score(track_i, tracks[j])
                if score < 0.46:
                    continue
                if (
                    stats.get("duplicate_ratio", 0.0) < 0.42
                    and not (
                        stats.get("duplicate_ratio", 0.0) >= 0.34
                        and stats.get("overlap_fraction", 0.0) >= 0.45
                        and stats.get("mean_center_norm", 9.0) <= 0.55
                    )
                ):
                    continue
                vis_i = _track_visible_len(track_i)
                vis_j = _track_visible_len(tracks[j])
                mean_q_i = sum(float(v) for v in track_i.get("q_by_frame", {}).values()) / max(vis_i, 1)
                mean_q_j = sum(float(v) for v in tracks[j].get("q_by_frame", {}).values()) / max(vis_j, 1)
                if (vis_i, mean_q_i, -int(track_i.get("birth_frame", 0))) >= (
                    vis_j,
                    mean_q_j,
                    -int(tracks[j].get("birth_frame", 0)),
                ):
                    primary, secondary = i, j
                else:
                    primary, secondary = j, i
                if best is None or score > best[0]:
                    best = (score, primary, secondary, stats)

        if best is None:
            break

        score, primary, secondary, stats = best
        if primary in removed or secondary in removed:
            continue
        _merge_track_payload_inplace(tracks[primary], tracks[secondary])
        removed.add(secondary)
        if len(debug) < 200:
            debug.append({
                "primary": int(primary),
                "secondary": int(secondary),
                **stats,
            })

    return [track for idx, track in enumerate(tracks) if idx not in removed], debug


def _person_track_boundary_score(left: Dict, right: Dict) -> Tuple[float, Dict[str, float]]:
    left_frames = sorted(int(t) for t in left.get("mask_by_frame", {}).keys())
    right_frames = sorted(int(t) for t in right.get("mask_by_frame", {}).keys())
    if not left_frames or not right_frames:
        return -1.0, {}

    gap = int(right_frames[0]) - int(left_frames[-1])
    if gap < -6 or gap > 42:
        return -1.0, {}

    left_eval = left_frames[-3:]
    right_eval = right_frames[:3]
    best_pair: Optional[Tuple[float, int, int, float, float, float]] = None
    for left_t in left_eval:
        for right_t in right_eval:
            pair_gap = int(right_t) - int(left_t)
            if pair_gap < -6 or pair_gap > 42:
                continue
            left_box = left["box_by_frame"][left_t]
            right_box = right["box_by_frame"][right_t]
            box_iou = _box_iou_xyxy_torch(left_box, right_box)
            center_norm = _box_center_norm_torch(left_box, right_box)
            area_left = float(left["area_by_frame"][left_t])
            area_right = float(right["area_by_frame"][right_t])
            area_sim = min(area_left, area_right) / max(area_left, area_right, 1e-6)
            if center_norm > 1.15 and box_iou < 0.05:
                continue
            if area_sim < 0.16 and box_iou < 0.10:
                continue
            pair_quality = (
                0.45 * max(0.0, 1.0 - min(center_norm, 2.0) / 2.0)
                + 0.30 * box_iou
                + 0.25 * area_sim
            )
            if best_pair is None or pair_quality > best_pair[0]:
                best_pair = (pair_quality, int(left_t), int(right_t), float(box_iou), float(center_norm), float(area_sim))
    if best_pair is None:
        return -1.0, {}

    _pair_quality, left_t, right_t, box_iou, center_norm, area_sim = best_pair
    duration_bonus = min(
        len(left.get("mask_by_frame", {})),
        len(right.get("mask_by_frame", {})),
        30,
    ) / 30.0
    gap_penalty = max(float(gap), 0.0) / 42.0
    score = (
        0.40 * max(0.0, 1.0 - min(center_norm, 2.0) / 2.0)
        + 0.28 * box_iou
        + 0.24 * area_sim
        + 0.10 * duration_bonus
        - 0.12 * gap_penalty
    )
    return float(score), {
        "gap": float(gap),
        "left_frame": float(left_t),
        "right_frame": float(right_t),
        "box_iou": float(box_iou),
        "center_norm": float(center_norm),
        "area_sim": float(area_sim),
        "score": float(score),
    }


def _merge_adjacent_efficient_person_tracks(tracks: List[Dict]) -> Tuple[List[Dict], List[Dict[str, object]]]:
    tracks = list(tracks)
    removed: set[int] = set()
    debug: List[Dict[str, object]] = []

    while True:
        best: Optional[Tuple[float, int, int, Dict[str, float]]] = None
        for i, left in enumerate(tracks):
            if i in removed or not _is_person_track(left):
                continue
            left_frames = sorted(int(t) for t in left.get("mask_by_frame", {}).keys())
            if not left_frames:
                continue
            for j, right in enumerate(tracks):
                if i == j or j in removed or not _is_person_track(right):
                    continue
                right_frames = sorted(int(t) for t in right.get("mask_by_frame", {}).keys())
                if not right_frames:
                    continue
                if left_frames[-1] > right_frames[0] + 4:
                    continue
                score, stats = _person_track_boundary_score(left, right)
                if score < 0.41:
                    continue
                if stats.get("center_norm", 9.0) > 0.95 and stats.get("box_iou", 0.0) < 0.08:
                    continue
                if best is None or score > best[0]:
                    best = (score, i, j, stats)

        if best is None:
            break

        score, primary, secondary, stats = best
        if primary in removed or secondary in removed:
            continue
        _merge_track_payload_inplace(tracks[primary], tracks[secondary])
        removed.add(secondary)
        if len(debug) < 200:
            debug.append({
                "primary": int(primary),
                "secondary": int(secondary),
                **stats,
            })

    return [track for idx, track in enumerate(tracks) if idx not in removed], debug


def _postprocess_efficient_sparse_output(masklet_output: SparseMaskletOutput) -> SparseMaskletOutput:
    """EfficientSAM3 is high-recall; prune only clear fragments after mature merge.

    We keep detector-backed people that do not overlap stronger tracks, but
    remove tiny one-off person shards and short fragments that are mostly
    contained by a longer person track. This keeps the recall improvements from
    sparse person support without flooding downstream code with duplicate IDs.
    """
    tracks = list(masklet_output.tracks)
    if not tracks:
        return masklet_output

    tracks, human_label_canonicalized = _canonicalize_efficient_human_labels(tracks)
    tracks, structure_merge_debug = _merge_adjacent_efficient_structure_tracks(tracks)
    tracks, structure_prune_debug = _prune_weak_efficient_structure_tracks(tracks)
    tracks, stuff_prune_debug = _prune_excess_efficient_stuff_tracks(tracks)
    tracks, structure_semantic_debug = _regularize_efficient_structure_semantics(tracks)
    tracks, structure_post_semantic_merge_debug = _merge_adjacent_efficient_structure_tracks(tracks)
    tracks, temporal_repair_debug = _repair_efficient_person_temporal_collapses(tracks)
    tracks, long_gap_repair_debug = _repair_efficient_person_long_gaps(
        tracks,
        total_frames=int(masklet_output.num_frames),
    )
    tracks, overlap_merge_debug = _merge_overlapping_efficient_person_tracks(tracks)
    tracks, wide_person_drift_debug = _prune_wide_person_drift_frames(tracks)
    tracks, frame_suppress_debug = _suppress_efficient_duplicate_person_frames(
        tracks,
        total_frames=int(masklet_output.num_frames),
    )
    tracks, merge_debug = _merge_adjacent_efficient_person_tracks(tracks)
    tracks, edge_prune_debug = _prune_edge_stuck_efficient_person_tracks(tracks)

    removed: set[int] = set()
    debug: List[Dict[str, object]] = []
    person_indices = [idx for idx, track in enumerate(tracks) if _is_person_track(track)]
    person_indices.sort(
        key=lambda idx: (
            -(
                len(tracks[idx].get("mask_by_frame", {}))
                * max(_efficient_track_mean_area(tracks[idx]), 1e-4) ** 0.5
            ),
            -_efficient_track_mean_area(tracks[idx]),
            -_efficient_track_max_area(tracks[idx]),
            -len(tracks[idx].get("mask_by_frame", {})),
            int(tracks[idx].get("birth_frame", 0)),
        )
    )

    stronger: List[int] = []
    for idx in person_indices:
        track = tracks[idx]
        visible = len(track.get("mask_by_frame", {}))
        mean_area = _efficient_track_mean_area(track)
        max_area = _efficient_track_max_area(track)

        drop_reason = None
        if visible <= 2:
            drop_reason = "tiny_person_very_short"
        elif visible <= 6:
            drop_reason = "person_too_short"
        elif visible <= 12 and mean_area < 0.090:
            drop_reason = "short_person_fragment"
        elif visible <= 18 and mean_area < 0.120:
            drop_reason = "short_person_fragment_any_area"
        elif visible <= 24 and mean_area < 0.035:
            drop_reason = "short_small_person_fragment"
        elif visible <= 40 and mean_area < 0.020 and max_area < 0.060:
            drop_reason = "short_low_area_person_fragment"
        elif visible <= 64 and mean_area < 0.012 and max_area < 0.040:
            drop_reason = "small_low_area_person_fragment"
        elif visible <= 4 and mean_area < 0.018 and max_area < 0.040:
            drop_reason = "tiny_person_short_low_area"
        elif visible <= 6 and mean_area < 0.010 and max_area < 0.030:
            drop_reason = "tiny_person_short"

        if (
            drop_reason is None
            and stronger
            and visible <= max(36, int(0.50 * int(masklet_output.num_frames)))
            and mean_area < 0.012
            and max_area < 0.025
        ):
            for ref_idx in stronger:
                ratio, checked = _efficient_person_duplicate_ratio(track, tracks[ref_idx])
                if checked >= 2 and ratio >= 0.32:
                    drop_reason = f"tiny_person_near_stronger:{ref_idx}"
                    break

        if drop_reason is None and stronger:
            shape_stats = _person_box_shape_stats(track)
            for ref_idx in stronger:
                ref_track = tracks[ref_idx]
                ref_visible = len(ref_track.get("mask_by_frame", {}))
                ref_mean = _efficient_track_mean_area(ref_track)
                if (
                    ref_visible >= max(48, int(0.40 * int(masklet_output.num_frames)))
                    and visible <= max(72, int(0.40 * int(masklet_output.num_frames)))
                    and mean_area < 0.012
                    and max_area < 0.030
                    and ref_mean >= max(4.0 * mean_area, 0.040)
                    and float(shape_stats.get("mean_bottom", 1.0)) < 0.72
                ):
                    drop_reason = f"small_non_floor_person_fragment:{ref_idx}"
                    break

        if drop_reason is None and stronger:
            for ref_idx in stronger:
                ref_track = tracks[ref_idx]
                ref_visible = len(ref_track.get("mask_by_frame", {}))
                ref_mean = _efficient_track_mean_area(ref_track)
                if (
                    ref_visible >= max(8, int(0.70 * int(masklet_output.num_frames)))
                    and ref_mean >= max(2.8 * mean_area, 0.045)
                    and visible <= int(0.90 * int(masklet_output.num_frames))
                    and mean_area < 0.014
                    and (max_area < 0.025 or mean_area < 0.004)
                ):
                    ratio, checked = _efficient_person_duplicate_ratio(track, ref_track)
                    if checked >= 3 and ratio >= 0.40:
                        drop_reason = f"small_person_fragment_near_dominant:{ref_idx}"
                        break
                if (
                    ref_visible >= max(40, 2 * max(visible, 1))
                    and ref_mean >= max(3.8 * mean_area, 0.045)
                    and abs(int(track.get("birth_frame", 0)) - int(ref_track.get("birth_frame", 0))) <= 4
                    and visible <= max(40, int(0.35 * int(masklet_output.num_frames)))
                    and mean_area < 0.014
                    and max_area < 0.025
                ):
                    ratio, checked = _efficient_person_duplicate_ratio(track, ref_track)
                    if checked >= 2 and ratio >= 0.45:
                        drop_reason = f"small_person_fragment_same_birth:{ref_idx}"
                        break
                if (
                    ref_visible >= max(12, int(0.85 * int(masklet_output.num_frames)))
                    and ref_mean >= max(6.0 * mean_area, 0.055)
                    and abs(int(track.get("birth_frame", 0)) - int(ref_track.get("birth_frame", 0))) <= 4
                    and visible <= int(0.60 * int(masklet_output.num_frames))
                    and mean_area < 0.012
                    and max_area < 0.022
                ):
                    drop_reason = f"tiny_person_shadow_of_dominant:{ref_idx}"
                    break

        if drop_reason is None and (visible <= 36 or mean_area < 0.018):
            for ref_idx in stronger:
                ratio, checked = _efficient_person_duplicate_ratio(track, tracks[ref_idx])
                if checked >= 2 and ratio >= (0.45 if visible <= 20 else 0.58):
                    drop_reason = f"duplicate_person_fragment:{ref_idx}"
                    break

        if drop_reason is not None:
            removed.add(idx)
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": str(track.get("L_sem", "")),
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "max_area": float(max_area),
                    "reason": drop_reason,
                })
            continue
        stronger.append(idx)

    for idx, track in enumerate(tracks):
        if idx in removed:
            continue
        if _track_visible_len(track) <= 0:
            removed.add(idx)
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": str(track.get("L_sem", "")),
                    "visible_frames": 0,
                    "mean_area": 0.0,
                    "reason": "empty_after_frame_suppression",
                })
            continue
        if _is_person_track(track):
            continue
        visible = len(track.get("mask_by_frame", {}))
        mean_area = _efficient_track_mean_area(track)
        max_area = _efficient_track_max_area(track)
        label_l = str(track.get("L_sem", "")).strip().lower()
        source_type = str(track.get("source_type", ""))
        sem_group = int(track.get("G_sem", -1))
        drop_reason = None

        if label_l == "book":
            drop_reason = "disabled_book_label"

        if source_type == "thing_tracked" and sem_group == SEMANTIC_GROUP_STATIC_THING:
            if visible <= 1:
                drop_reason = "single_frame_static_false_positive"
            elif label_l in {"door", "desk", "table", "window"}:
                drop_reason = "disabled_static_prompt_label"
            elif visible <= 4 and mean_area < 0.010:
                drop_reason = "tiny_static_very_short"
            elif visible <= 20 and mean_area > 0.080:
                drop_reason = "short_static_large_drift"
            elif (
                label_l in {"door", "window"}
                and mean_area >= 0.14
                and max_area >= 0.22
                and visible <= max(8, int(0.92 * int(masklet_output.num_frames)))
            ):
                drop_reason = "large_background_static_drift"
            elif visible <= 24 and mean_area < 0.006:
                drop_reason = "tiny_static_short"
            elif label_l in {"desk", "door"} and visible <= 90 and mean_area < 0.008:
                drop_reason = "tiny_static_low_value"

        if drop_reason is None and source_type == "thing_tracked" and sem_group == SEMANTIC_GROUP_MOVABLE_THING:
            low_value_label = label_l in {"book", "chair", "bag", "bottle", "cup", "phone", "mouse", "keyboard"}
            if visible <= 4 and mean_area < 0.010:
                drop_reason = "tiny_movable_very_short"
            elif low_value_label and visible <= 18 and mean_area < 0.010:
                drop_reason = "low_value_movable_short"
            elif low_value_label and mean_area < 0.0022 and max_area < 0.0065:
                drop_reason = "low_value_movable_tiny"
            elif label_l == "chair" and mean_area < 0.0045 and max_area < 0.011:
                drop_reason = "tiny_chair_fragment"
            elif label_l == "chair" and mean_area < 0.0040 and max_area < 0.018:
                drop_reason = "tiny_long_chair_fragment"
            elif label_l == "book" and mean_area < 0.0020 and max_area < 0.006:
                drop_reason = "tiny_book_fragment"
            elif label_l == "book" and mean_area < 0.0040 and max_area < 0.018:
                drop_reason = "tiny_long_book_fragment"

        if drop_reason is not None:
            removed.add(idx)
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": str(track.get("L_sem", "")),
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "max_area": float(max_area),
                    "reason": drop_reason,
                })
            continue

        if (
            track.get("source_type") == "thing_tracked"
            and int(track.get("G_sem", -1)) == SEMANTIC_GROUP_MOVABLE_THING
            and visible <= 2
            and mean_area < 0.0025
        ):
            removed.add(idx)
            if len(debug) < 200:
                debug.append({
                    "track_index": int(idx),
                    "label": str(track.get("L_sem", "")),
                    "visible_frames": int(visible),
                    "mean_area": float(mean_area),
                    "reason": "tiny_movable_very_short",
                })

    working_tracks = [
        track for idx, track in enumerate(tracks)
        if idx not in removed
    ]
    working_tracks, structure_occlusion_debug = _subtract_thing_masks_from_structure_tracks(
        working_tracks
    )
    working_tracks, structure_geometry_frame_debug = _clean_structure_geometry_frames(
        working_tracks
    )
    working_tracks = [track for track in working_tracks if _track_visible_len(track) > 0]

    if not removed:
        masklet_output.debug["efficient_fragment_prune_debug"] = []
        masklet_output.debug["efficient_human_label_canonicalized"] = int(human_label_canonicalized)
        masklet_output.debug["efficient_structure_merge_debug"] = structure_merge_debug
        masklet_output.debug["efficient_structure_merge_count"] = int(len(structure_merge_debug))
        masklet_output.debug["efficient_structure_post_semantic_merge_debug"] = structure_post_semantic_merge_debug
        masklet_output.debug["efficient_structure_post_semantic_merge_count"] = int(len(structure_post_semantic_merge_debug))
        masklet_output.debug["efficient_structure_prune_debug"] = structure_prune_debug
        masklet_output.debug["efficient_stuff_prune_debug"] = stuff_prune_debug
        masklet_output.debug["efficient_stuff_prune_removed"] = int(len(stuff_prune_debug))
        masklet_output.debug["efficient_structure_semantic_debug"] = structure_semantic_debug
        masklet_output.debug["efficient_structure_occlusion_subtract_debug"] = structure_occlusion_debug
        masklet_output.debug["efficient_structure_geometry_frame_debug"] = structure_geometry_frame_debug
        masklet_output.debug["efficient_person_temporal_repair_debug"] = temporal_repair_debug
        masklet_output.debug["efficient_person_temporal_repair_count"] = int(len(temporal_repair_debug))
        masklet_output.debug["efficient_person_long_gap_repair_debug"] = long_gap_repair_debug
        masklet_output.debug["efficient_person_long_gap_repair_count"] = int(len(long_gap_repair_debug))
        masklet_output.debug["efficient_person_overlap_merge_debug"] = overlap_merge_debug
        masklet_output.debug["efficient_person_overlap_merge_count"] = int(len(overlap_merge_debug))
        masklet_output.debug["efficient_wide_person_drift_debug"] = wide_person_drift_debug
        masklet_output.debug["efficient_wide_person_drift_removed"] = int(
            max(0, len(wide_person_drift_debug) - 1)
        )
        masklet_output.debug["efficient_frame_suppress_debug"] = frame_suppress_debug
        masklet_output.debug["efficient_person_merge_debug"] = merge_debug
        masklet_output.debug["efficient_person_merge_count"] = int(len(merge_debug))
        masklet_output.debug["efficient_edge_person_prune_debug"] = edge_prune_debug
        masklet_output.debug["efficient_edge_person_prune_removed"] = int(len(edge_prune_debug))
        masklet_output.tracks = working_tracks
        masklet_output.num_masklets = len(working_tracks)
        masklet_output.debug["J_thing"] = sum(1 for track in working_tracks if track["source_type"] == "thing_tracked")
        masklet_output.debug["J_structure"] = sum(1 for track in working_tracks if track["source_type"] == "structure_tracked")
        masklet_output.debug["J_stuff"] = sum(1 for track in working_tracks if track["source_type"] == "stuff_static")
        return masklet_output

    kept_tracks = working_tracks
    masklet_output.tracks = kept_tracks
    masklet_output.num_masklets = len(kept_tracks)
    masklet_output.debug["efficient_human_label_canonicalized"] = int(human_label_canonicalized)
    masklet_output.debug["efficient_person_overlap_merge_debug"] = overlap_merge_debug
    masklet_output.debug["efficient_person_overlap_merge_count"] = int(len(overlap_merge_debug))
    masklet_output.debug["efficient_wide_person_drift_debug"] = wide_person_drift_debug
    masklet_output.debug["efficient_wide_person_drift_removed"] = int(
        max(0, len(wide_person_drift_debug) - 1)
    )
    masklet_output.debug["efficient_person_merge_debug"] = merge_debug
    masklet_output.debug["efficient_person_merge_count"] = int(len(merge_debug))
    masklet_output.debug["efficient_edge_person_prune_debug"] = edge_prune_debug
    masklet_output.debug["efficient_edge_person_prune_removed"] = int(len(edge_prune_debug))
    masklet_output.debug["efficient_structure_merge_debug"] = structure_merge_debug
    masklet_output.debug["efficient_structure_merge_count"] = int(len(structure_merge_debug))
    masklet_output.debug["efficient_structure_post_semantic_merge_debug"] = structure_post_semantic_merge_debug
    masklet_output.debug["efficient_structure_post_semantic_merge_count"] = int(len(structure_post_semantic_merge_debug))
    masklet_output.debug["efficient_structure_prune_debug"] = structure_prune_debug
    masklet_output.debug["efficient_stuff_prune_debug"] = stuff_prune_debug
    masklet_output.debug["efficient_stuff_prune_removed"] = int(len(stuff_prune_debug))
    masklet_output.debug["efficient_structure_semantic_debug"] = structure_semantic_debug
    masklet_output.debug["efficient_structure_occlusion_subtract_debug"] = structure_occlusion_debug
    masklet_output.debug["efficient_structure_geometry_frame_debug"] = structure_geometry_frame_debug
    masklet_output.debug["efficient_person_temporal_repair_debug"] = temporal_repair_debug
    masklet_output.debug["efficient_person_temporal_repair_count"] = int(len(temporal_repair_debug))
    masklet_output.debug["efficient_person_long_gap_repair_debug"] = long_gap_repair_debug
    masklet_output.debug["efficient_person_long_gap_repair_count"] = int(len(long_gap_repair_debug))
    masklet_output.debug["efficient_frame_suppress_debug"] = frame_suppress_debug
    masklet_output.debug["efficient_fragment_prune_debug"] = debug
    masklet_output.debug["efficient_fragment_prune_removed"] = int(len(removed))
    masklet_output.debug["J_thing"] = sum(1 for track in kept_tracks if track["source_type"] == "thing_tracked")
    masklet_output.debug["J_structure"] = sum(1 for track in kept_tracks if track["source_type"] == "structure_tracked")
    masklet_output.debug["J_stuff"] = sum(1 for track in kept_tracks if track["source_type"] == "stuff_static")
    return masklet_output


def finalize_global_tracks(
    global_tracks: List[Dict],
    total_frames: int,
    H: int,
    W: int,
    chunk_match_debug: List[Dict[str, object]],
) -> SparseMaskletOutput:
    # The mature SAM3.1 runner performs several expensive full-track cleaning
    # passes. EfficientSAM3 produces many short detector-support fragments, and
    # running the mature O(N^2) + per-mask cleanup over a full Taylor video can
    # dominate the end-to-end time. Keep the lightweight sparse representation
    # here, then run the efficient-specific person merge/prune pass below.
    masklet_output = SparseMaskletOutput(
        tracks=list(global_tracks),
        num_masklets=len(global_tracks),
        num_frames=total_frames,
        frame_height=H,
        frame_width=W,
        debug={
            "chunk_match_debug": chunk_match_debug,
            "num_chunks": len(chunk_match_debug),
            "efficient_fast_finalize": True,
            "J_thing": sum(1 for track in global_tracks if track["source_type"] == "thing_tracked"),
            "J_structure": sum(1 for track in global_tracks if track["source_type"] == "structure_tracked"),
            "J_stuff": sum(1 for track in global_tracks if track["source_type"] == "stuff_static"),
        },
    )
    return _postprocess_efficient_sparse_output(masklet_output)


def _default_checkpoint_path(output_pt: Optional[str]) -> Optional[str]:
    if not output_pt:
        return None
    stem, ext = os.path.splitext(output_pt)
    if ext:
        return f"{stem}.checkpoint{ext}"
    return f"{output_pt}.checkpoint.pt"


def _atomic_torch_save(payload: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _serialize_sparse_tracks(masklet_output: SparseMaskletOutput) -> List[Dict]:
    sparse_tracks = []
    for track in masklet_output.tracks:
        frames = sorted(int(t) for t in track["mask_by_frame"].keys())
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
    return sparse_tracks


def _save_sparse_masklet_pt(masklet_output: SparseMaskletOutput, path: str) -> None:
    _atomic_torch_save({
        "format": "sparse_masklets_v1",
        "frame_height": masklet_output.frame_height,
        "frame_width": masklet_output.frame_width,
        "num_masklets": masklet_output.num_masklets,
        "num_frames": masklet_output.num_frames,
        "tracks": _serialize_sparse_tracks(masklet_output),
        "debug": masklet_output.debug,
    }, path)


def _augment_with_segformer_stuff_tracks(
    masklet_output: SparseMaskletOutput,
    image_paths: List[str],
    *,
    model_name: str,
    stride: int,
    batch_size: int,
    labels: List[str],
    device: str,
    replace_existing: bool,
) -> SparseMaskletOutput:
    if not model_name or stride <= 0 or not labels:
        return masklet_output

    from PIL import Image
    from torch.nn import functional as F
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

    H = int(masklet_output.frame_height)
    W = int(masklet_output.frame_width)
    total_frames = int(masklet_output.num_frames)
    if H <= 0 or W <= 0 or total_frames <= 0:
        return masklet_output

    wanted = {str(label).strip().lower() for label in labels if str(label).strip()}
    alias_to_canonical = {
        "wall": "wall",
        "building": "building",
        "floor": "floor",
        "ceiling": "ceiling",
        "road": "road",
        "sidewalk": "sidewalk",
        "stairs": "stair",
        "stairway": "stair",
        "stair": "stair",
    }

    print(
        f"Loading SegFormer STUFF model ({model_name}) stride={stride} labels={sorted(wanted)} ...",
        flush=True,
    )
    t0 = time.time()
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    seg_device = device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model = model.to(seg_device).eval()
    id_to_canonical: Dict[int, str] = {}
    for raw_id, raw_label in model.config.id2label.items():
        canonical = alias_to_canonical.get(str(raw_label).strip().lower())
        if canonical and canonical in wanted:
            id_to_canonical[int(raw_id)] = canonical
    if not id_to_canonical:
        print("SegFormer STUFF: no requested labels are present in model labels; skipping.")
        return masklet_output

    if replace_existing:
        before = len(masklet_output.tracks)
        masklet_output.tracks = [
            track for track in masklet_output.tracks
            if str(track.get("source_type", "")) not in {"structure_tracked", "stuff_static"}
        ]
        masklet_output.debug["segformer_stuff_replaced_existing"] = int(before - len(masklet_output.tracks))

    label_tracks: Dict[str, Dict] = {}
    for label in sorted(set(id_to_canonical.values())):
        label_tracks[label] = {
            "mask_by_frame": {},
            "box_by_frame": {},
            "q_by_frame": {},
            "area_by_frame": {},
            "L_sem": label,
            "G_sem": SEMANTIC_GROUP_STRUCTURE_ANCHOR,
            "W_sem": 1.0,
            "source_type": "stuff_static",
            "birth_frame": 0,
            "frame_height": H,
            "frame_width": W,
        }

    sample_frames = list(range(0, total_frames, max(int(stride), 1)))
    if sample_frames[-1] != total_frames - 1:
        sample_frames.append(total_frames - 1)
    batch_size = max(int(batch_size), 1)
    frames_with_any_stuff = 0
    masks_added = 0

    ids_by_label: Dict[str, List[int]] = {}
    for class_id, canonical in id_to_canonical.items():
        ids_by_label.setdefault(canonical, []).append(int(class_id))

    def _geometry_filter(label: str, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(bool)
        if label in {"floor", "road", "sidewalk", "stair"}:
            mask[: int(0.34 * H), :] = False
        elif label == "ceiling":
            mask[int(0.48 * H):, :] = False
        elif label in {"wall", "building"}:
            mask[: int(0.03 * H), :] = False
            mask[int(0.94 * H):, :] = False
        return mask

    def _mask_is_plausible(label: str, mask: np.ndarray) -> bool:
        area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
        if area_ratio < 0.004:
            return False
        max_area = {
            "wall": 0.70,
            "building": 0.70,
            "floor": 0.55,
            "ceiling": 0.45,
            "stair": 0.45,
            "road": 0.55,
            "sidewalk": 0.55,
        }.get(label, 0.50)
        if area_ratio > max_area:
            return False
        box = _mask_to_box_tensor(mask)
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        y_center = (y1 + y2) / max(2.0 * float(H), 1.0)
        if label == "ceiling" and y_center > 0.34:
            return False
        if label in {"floor", "road", "sidewalk", "stair"} and y_center < 0.48:
            return False
        return True

    with torch.inference_mode():
        for batch_start in range(0, len(sample_frames), batch_size):
            batch_frames = sample_frames[batch_start: batch_start + batch_size]
            pil_images = [
                Image.open(image_paths[int(frame_idx)]).convert("RGB")
                for frame_idx in batch_frames
            ]
            inputs = processor(images=pil_images, return_tensors="pt").to(seg_device)
            logits = model(**inputs).logits
            logits = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            for local_idx, sample_t in enumerate(batch_frames):
                pred = preds[local_idx]
                next_sample = (
                    sample_frames[batch_start + local_idx + 1]
                    if batch_start + local_idx + 1 < len(sample_frames)
                    else total_frames
                )
                filled_until = min(int(next_sample), total_frames)
                label_masks: Dict[str, np.ndarray] = {}
                for label, class_ids in ids_by_label.items():
                    mask = np.isin(pred, class_ids)
                    mask = _geometry_filter(label, mask)
                    if not _mask_is_plausible(label, mask):
                        continue
                    label_masks[label] = mask
                if not label_masks:
                    continue
                frames_with_any_stuff += 1
                for label, mask in label_masks.items():
                    packed = _pack_mask_np(mask)
                    box = _mask_to_box_tensor(mask)
                    area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
                    track = label_tracks[label]
                    for t in range(int(sample_t), filled_until):
                        track["mask_by_frame"][int(t)] = packed
                        track["box_by_frame"][int(t)] = box
                        track["q_by_frame"][int(t)] = 0.70
                        track["area_by_frame"][int(t)] = area_ratio
                        masks_added += 1

            if (batch_start // batch_size + 1) % 10 == 0:
                print(
                    f"  SegFormer STUFF sampled {min(batch_start + batch_size, len(sample_frames))}/"
                    f"{len(sample_frames)} frames",
                    flush=True,
                )

    added_tracks = [track for track in label_tracks.values() if track["mask_by_frame"]]
    masklet_output.tracks.extend(added_tracks)
    masklet_output.num_masklets = len(masklet_output.tracks)
    masklet_output.debug["segformer_stuff_model"] = model_name
    masklet_output.debug["segformer_stuff_stride"] = int(stride)
    masklet_output.debug["segformer_stuff_sample_frames"] = int(len(sample_frames))
    masklet_output.debug["segformer_stuff_frames_with_any"] = int(frames_with_any_stuff)
    masklet_output.debug["segformer_stuff_tracks_added"] = int(len(added_tracks))
    masklet_output.debug["segformer_stuff_masks_added"] = int(masks_added)
    masklet_output.debug["segformer_stuff_elapsed_seconds"] = float(time.time() - t0)
    masklet_output.debug["J_thing"] = sum(1 for track in masklet_output.tracks if track["source_type"] == "thing_tracked")
    masklet_output.debug["J_structure"] = sum(1 for track in masklet_output.tracks if track["source_type"] == "structure_tracked")
    masklet_output.debug["J_stuff"] = sum(1 for track in masklet_output.tracks if track["source_type"] == "stuff_static")
    print(
        "SegFormer STUFF added "
        f"{len(added_tracks)} track(s), {masks_added} frame masks in {time.time() - t0:.1f}s",
        flush=True,
    )
    return masklet_output


def _save_global_checkpoint(
    path: str,
    *,
    next_chunk_idx: int,
    global_tracks: List[Dict],
    chunk_match_debug: List[Dict[str, object]],
    total_frames: int,
    frame_height: int,
    frame_width: int,
    chunks: List[Tuple[int, int]],
    args: argparse.Namespace,
) -> None:
    _atomic_torch_save({
        "format": "efficient_sparse_global_checkpoint_v1",
        "next_chunk_idx": int(next_chunk_idx),
        "global_tracks": global_tracks,
        "chunk_match_debug": chunk_match_debug,
        "num_frames": int(total_frames),
        "frame_height": int(frame_height),
        "frame_width": int(frame_width),
        "chunks": [(int(s), int(e)) for s, e in chunks],
        "chunk_size": int(args.chunk_size),
        "chunk_overlap": int(args.chunk_overlap),
        "start_frame": int(args.start_frame),
        "end_frame": int(args.end_frame),
        "stride": int(args.stride),
        "input": str(args.input),
    }, path)


def _load_global_checkpoint(
    path: str,
    *,
    total_frames: int,
    frame_height: int,
    frame_width: int,
    chunks: List[Tuple[int, int]],
) -> Tuple[int, List[Dict], List[Dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format") != "efficient_sparse_global_checkpoint_v1":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    if int(payload.get("num_frames", -1)) != int(total_frames):
        raise ValueError(
            f"Checkpoint frame count mismatch: {payload.get('num_frames')} vs {total_frames}"
        )
    if int(payload.get("frame_height", -1)) != int(frame_height) or int(payload.get("frame_width", -1)) != int(frame_width):
        raise ValueError(
            "Checkpoint frame size mismatch: "
            f"{payload.get('frame_height')}x{payload.get('frame_width')} vs {frame_height}x{frame_width}"
        )
    saved_chunks = [(int(s), int(e)) for s, e in payload.get("chunks", [])]
    if saved_chunks != [(int(s), int(e)) for s, e in chunks]:
        raise ValueError("Checkpoint chunk schedule mismatch; refusing to resume stale state.")
    return (
        int(payload.get("next_chunk_idx", 0)),
        list(payload.get("global_tracks", [])),
        list(payload.get("chunk_match_debug", [])),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if args.tracker_backend in {"efficientsam3", "efficient"} and args.efficientsam3_checkpoint and not os.path.exists(args.efficientsam3_checkpoint):
        sys.exit(f"--efficientsam3_checkpoint not found: {args.efficientsam3_checkpoint}")
    if args.tracker_backend == "cutie" and args.sam3_checkpoint and not os.path.exists(args.sam3_checkpoint):
        sys.exit(f"--sam3_checkpoint not found: {args.sam3_checkpoint}")
    if args.tracker_backend == "edgetam" and args.edgetam_checkpoint and not (
        os.path.exists(args.edgetam_checkpoint)
        or os.path.exists(os.path.join(REPO_ROOT, args.edgetam_checkpoint))
    ):
        sys.exit(f"--edgetam_checkpoint not found: {args.edgetam_checkpoint}")
    if not args.device.startswith("cuda"):
        sys.exit("--device must be a CUDA device for efficient video inference")

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
        checkpoint_path = args.checkpoint_pt or _default_checkpoint_path(args.output_pt)
        if checkpoint_path and args.checkpoint_every_chunks > 0:
            print(f"Chunk checkpoint: {checkpoint_path}")

        # Build kwargs
        frontend_kwargs: dict = dict(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            ann_frame_idx=args.ann_frame_idx,
            discovery_frame_stride=args.discovery_frame_stride,
            max_thing_objects=args.max_thing_objects,
            sam31_text_track_labels=args.sam31_text_track_labels,
            sam31_structure_prompt_labels=args.sam31_structure_prompt_labels,
            sam31_structure_prompt_frame_count=args.sam31_structure_prompt_frame_count,
            sam31_structure_prompt_chunk_stride=args.sam31_structure_prompt_chunk_stride,
            sam31_person_refresh_prompt_frames=args.sam31_person_refresh_prompt_frames,
            sam31_nontext_object_prompt_budget=args.sam31_nontext_object_prompt_budget,
            sam31_nontext_object_prompt_min_support=args.sam31_nontext_object_prompt_min_support,
            sam31_nontext_sparse_support=bool(args.sam31_nontext_sparse_support),
            sam31_max_text_prompt_objects=args.sam31_max_text_prompt_objects,
            sam31_max_movable_objects=args.sam31_max_movable_objects,
            sam31_max_static_objects=args.sam31_max_static_objects,
            sam31_max_structure_objects=args.sam31_max_structure_objects,
            cutie_max_internal_size=args.cutie_max_internal_size,
            sam3_cutie_sam_confidence_threshold=args.sam3_cutie_sam_confidence_threshold,
            sam3_cutie_detection_frame_count=args.sam3_cutie_detection_frame_count,
            sam3_cutie_max_prompt_dets_per_label=args.sam3_cutie_max_prompt_dets_per_label,
            sam3_cutie_use_yoloe=bool(args.sam3_cutie_use_yoloe),
        )
        if args.thing_prompts:
            frontend_kwargs["thing_prompts"] = [s.strip() for s in args.thing_prompts.split(",")]
        if args.stuff_prompts:
            frontend_kwargs["stuff_prompts"] = [s.strip() for s in args.stuff_prompts.split(",")]

        global_tracks: List[Dict] = []
        chunk_match_debug: List[Dict[str, object]] = []
        resume_chunk_idx = 0
        if args.resume_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
            resume_chunk_idx, global_tracks, chunk_match_debug = _load_global_checkpoint(
                checkpoint_path,
                total_frames=total_frames,
                frame_height=sample_h,
                frame_width=sample_w,
                chunks=chunks,
            )
            resume_chunk_idx = max(0, min(int(resume_chunk_idx), len(chunks)))
            print(
                f"Resumed checkpoint at chunk {resume_chunk_idx}/{len(chunks)} "
                f"with {len(global_tracks)} global tracks."
            )
        elif args.resume_checkpoint:
            print("Resume requested but no checkpoint was found; starting from scratch.")

        # Run Stage C chunk-by-chunk
        print("Running Video Masklet Front-end (Stage C) ...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        if resume_chunk_idx < len(chunks):
            # Build frontend only when there is actual video inference left to do.
            if args.tracker_backend == "sam2":
                print("Loading models (SAM3 + YOLOE + SAM2.1) ...")
            elif args.tracker_backend == "edgetam":
                print("Loading models (SAM3 + YOLOE + EdgeTAM) ...")
            elif args.tracker_backend == "cutie":
                print("Loading models (SAM3 + Cutie) ...")
            else:
                print("Loading models (EfficientSAM3 + YOLOE) ...")
            model_t0 = time.time()

            build_kwargs: dict = dict(
                device=args.device,
                tracker_backend=args.tracker_backend,
                sam3_checkpoint=args.sam3_checkpoint,
                sam2_checkpoint=args.sam2_checkpoint,
                sam2_model_cfg=args.sam2_model_cfg,
                edgetam_checkpoint=args.edgetam_checkpoint,
                edgetam_model_cfg=args.edgetam_model_cfg,
                efficientsam3_checkpoint=args.efficientsam3_checkpoint,
                efficientsam3_repo_id=args.efficientsam3_repo_id,
                efficientsam3_filename=args.efficientsam3_filename,
                efficientsam3_cache_dir=args.efficientsam3_cache_dir,
                efficientsam3_backbone_type=args.efficientsam3_backbone_type,
                efficientsam3_model_name=args.efficientsam3_model_name,
                efficientsam3_text_encoder_type=args.efficientsam3_text_encoder_type,
                efficientsam3_text_context_length=args.efficientsam3_text_context_length,
                yoloe_model=args.yoloe_model,
            )

            frontend = _run_with_heartbeat(
                "  Loading models",
                lambda: EfficientVideoMaskletFrontend.from_config(**build_kwargs, **frontend_kwargs),
            )
            print(f"Models loaded in {time.time() - model_t0:.1f}s")

            chunks_processed_this_run = 0
            for ci, (start, end) in enumerate(chunks[resume_chunk_idx:], start=resume_chunk_idx):
                seed_detections_by_frame = None
                discovery_frame_indices = build_chunk_discovery_indices(
                    chunk_len=end - start,
                    ann_frame_idx=args.ann_frame_idx,
                    discovery_frame_stride=args.discovery_frame_stride,
                    overlap=args.chunk_overlap,
                    chunk_idx=ci,
                )
                if global_tracks and args.chunk_overlap > 0:
                    seed_detections_by_frame = _build_global_chunk_seed_detections(
                        global_tracks,
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
                    chunk_index=ci,
                )
                print(f"  Stage C chunk done in {time.time() - chunk_t0:.2f}s")
                if chunk_output.debug.get("discovery_frames") is not None:
                    print(f"  Discovery frames (local): {chunk_output.debug['discovery_frames']}")
                if bool(args.print_chunk_summaries):
                    print_masklet_output(chunk_output)
                else:
                    thing_count = sum(1 for s in chunk_output.source_type if s == "thing_tracked")
                    structure_count = sum(1 for s in chunk_output.source_type if s == "structure_tracked")
                    stuff_count = sum(1 for s in chunk_output.source_type if s == "stuff_static")
                    print(
                        "  Chunk summary: "
                        f"J={chunk_output.num_masklets}  "
                        f"thing={thing_count}  structure={structure_count}  stuff={stuff_count}",
                        flush=True,
                    )
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
                del chunk_output
                seed_detections_by_frame = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (
                    checkpoint_path
                    and args.checkpoint_every_chunks > 0
                    and (
                        ((ci + 1) % int(args.checkpoint_every_chunks) == 0)
                        or ci == len(chunks) - 1
                    )
                ):
                    print(f"  Saving chunk checkpoint at next_chunk_idx={ci + 1} ...", flush=True)
                    _save_global_checkpoint(
                        checkpoint_path,
                        next_chunk_idx=ci + 1,
                        global_tracks=global_tracks,
                        chunk_match_debug=chunk_match_debug,
                        total_frames=total_frames,
                        frame_height=sample_h,
                        frame_width=sample_w,
                        chunks=chunks,
                        args=args,
                    )
                    print(f"  Saved chunk checkpoint at next_chunk_idx={ci + 1}")
                chunks_processed_this_run += 1
                if args.max_chunks_this_run > 0 and chunks_processed_this_run >= int(args.max_chunks_this_run):
                    if checkpoint_path:
                        print(f"  Saving stop checkpoint at next_chunk_idx={ci + 1} ...", flush=True)
                        _save_global_checkpoint(
                            checkpoint_path,
                            next_chunk_idx=ci + 1,
                            global_tracks=global_tracks,
                            chunk_match_debug=chunk_match_debug,
                            total_frames=total_frames,
                            frame_height=sample_h,
                            frame_width=sample_w,
                            chunks=chunks,
                            args=args,
                        )
                        print(f"  Saved stop checkpoint at next_chunk_idx={ci + 1}")
                    print(
                        f"Stopped after {chunks_processed_this_run} chunk(s); "
                        "resume with --resume_checkpoint to continue."
                    )
                    return
        else:
            print("All chunks are already present in the checkpoint; finalizing only.")

        masklet_output = finalize_global_tracks(
            global_tracks,
            total_frames=total_frames,
            H=sample_h,
            W=sample_w,
            chunk_match_debug=chunk_match_debug,
        )
        segformer_labels = [
            s.strip() for s in str(args.segformer_stuff_labels).split(",")
            if s.strip()
        ]
        masklet_output = _augment_with_segformer_stuff_tracks(
            masklet_output,
            image_paths,
            model_name=str(args.segformer_stuff_model).strip(),
            stride=int(args.segformer_stuff_stride),
            batch_size=int(args.segformer_stuff_batch_size),
            labels=segformer_labels,
            device=args.device,
            replace_existing=bool(args.segformer_stuff_replace_existing),
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_c_elapsed = float(time.time() - t0)
        print(f"\nStage C total done in {stage_c_elapsed:.2f}s")
        peak_alloc_gb: Optional[float] = None
        peak_reserved_gb: Optional[float] = None
        if torch.cuda.is_available():
            peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(
                f"Peak CUDA memory: allocated={peak_alloc_gb:.2f} GiB  "
                f"reserved={peak_reserved_gb:.2f} GiB"
            )
        masklet_output.debug["stage_c_elapsed_seconds"] = stage_c_elapsed
        if peak_alloc_gb is not None:
            masklet_output.debug["peak_allocated_gib"] = float(peak_alloc_gb)
        if peak_reserved_gb is not None:
            masklet_output.debug["peak_reserved_gib"] = float(peak_reserved_gb)

        print_masklet_output(masklet_output)

        if args.output_pt:
            _save_sparse_masklet_pt(masklet_output, args.output_pt)
            print(f"Saved masklet tensors to {args.output_pt}")

        create_tracking_video(
            image_paths, masklet_output, args.output_video,
            fps=args.fps, mask_alpha=args.mask_alpha,
            save_frames_dir=args.save_frames,
        )

    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
