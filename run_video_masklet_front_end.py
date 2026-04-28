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
import contextlib
import gc
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from natsort import natsorted

GSAM2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grounded-SAM-2")
if GSAM2_ROOT not in sys.path:
    sys.path.insert(0, GSAM2_ROOT)

from loger.pipeline.video_masklet_frontend import (
    DEFAULT_SEMANTIC_WEIGHTS,
    MaskletOutput,
    VideoMaskletFrontend,
    SEMANTIC_GROUP_NAMES,
    SEMANTIC_GROUP_MOVABLE_THING,
    SEMANTIC_GROUP_STATIC_THING,
    SEMANTIC_GROUP_STRUCTURE_ANCHOR,
    canonicalize_label,
    label_to_group,
    passes_structure_mask_quality,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
DEFAULT_EFFICIENTSAM3_REPO_ID = "Simon7108528/EfficientSAM3"
DEFAULT_EFFICIENTSAM3_FILENAME = "stage1_sam3p1/efficient_sam3p1_efficientvit_m_mobileclip_s0_ctx16.pt"
_CLIPSEG_STUFF_CACHE: Dict[Tuple[str, str], Tuple[object, torch.nn.Module]] = {}
_LSEG_STUFF_CACHE: Dict[Tuple[str, str, str, bool], torch.nn.Module] = {}


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
    p.add_argument("--processing_max_side", type=int, default=720,
                   help="Resize frames so the longest side is at most this value for Stage C inference, then upsample sparse masks back to the original frame size. Use 0 for full-resolution inference.")
    p.add_argument("--detector_max_side", type=int, default=0,
                   help="Optional separate detector input max side. 0 reuses the SAM/tracking frames; use 1280 to let YOLOE detect on original Taylor resolution while SAM tracks at --processing_max_side.")
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Frames per chunk (0 = all frames as one chunk).")
    p.add_argument("--chunk_overlap", type=int, default=4,
                   help="Overlap between consecutive chunks for cross-chunk ID matching.")
    p.add_argument("--seed_carry_gap", type=int, default=0,
                   help="If overlap masks are missing, reuse a recent THING mask as a weak handoff seed for up to N frames.")
    p.add_argument("--consolidate_primary_person", type=int, default=0,
                   help="Experimental single-subject postprocess: merge dominant person tracks into one ID. Default 0 avoids merging different people.")
    p.add_argument("--sam31_min_chunk_size", type=int, default=96,
                   help="Standalone SAM3.1 optimization: coalesce smaller requested chunks up to this size; 0 disables. Default 96 reduces object-slot pressure under the 9GB VRAM target.")
    p.add_argument("--sam31_route", default="custom", choices=["custom", "b_quality"],
                   help="Preset route. b_quality = SAM3.1 text-prompt detection/tracking as the primary THING path, 720px tracking, YOLOE-11l low-res assist.")

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
    p.add_argument("--sam31_offload_sam_during_detection", type=int, default=0,
                   help="SAM3.1 multiplex: move SAM to CPU while detector discovery runs to avoid stacked model VRAM (1/0).")
    p.add_argument("--sam31_enable_backward", type=int, default=0,
                   help="SAM3.1 multiplex: also run reverse propagation inside each chunk (1/0). Default 0 is faster and avoids reverse-cache failures.")
    p.add_argument("--sam31_text_track_labels", default="person",
                   help="Comma-separated labels that use SAM3.1 text-video tracking; use 'all' to text-track every detected label. Person aliases remain detector sparse support by default.")
    p.add_argument("--sam31_direct_text_prompt_labels", default="",
                   help="Comma-separated THING labels that SAM3.1 should query directly with text prompts, independent of YOLOE detections.")
    p.add_argument("--sam31_direct_text_prompt_frame_count", type=int, default=0,
                   help="Number of direct SAM3.1 text-prompt frames per chunk for --sam31_direct_text_prompt_labels. 0 disables direct text discovery.")
    p.add_argument("--sam31_structure_prompt_labels", default="",
                   help="Comma-separated structure labels directly queried with SAM3.1 text prompts for tracked structure/STUFF support.")
    p.add_argument("--sam31_structure_prompt_frame_count", type=int, default=1,
                   help="Number of discovery frames per chunk used for direct SAM3.1 structure prompt detection.")
    p.add_argument("--sam31_structure_prompt_chunk_stride", type=int, default=1,
                   help="Run direct SAM3.1 structure prompts every N chunks; 1 = every chunk, 0 = disabled.")
    p.add_argument("--sam31_person_refresh_prompt_frames", type=int, default=0,
                   help="Extra late-frame SAM3.1 text prompt refreshes for person labels per chunk. Default 0 relies on YOLOE sparse support to save SAM3.1 propagation time.")
    p.add_argument("--sam31_nontext_object_prompt_budget", type=int, default=0,
                   help="Max non-text-tracked YOLOE candidates to propagate with SAM3.1 object prompts per chunk; 0 keeps the default path fast.")
    p.add_argument("--sam31_nontext_object_prompt_min_support", type=int, default=2,
                   help="Minimum multi-frame detector support for non-seed non-text object prompts.")
    p.add_argument("--sam31_text_object_prompt_budget", type=int, default=0,
                   help="Max unmatched YOLOE candidates for SAM3.1 text-tracked labels, e.g. person, to force through SAM3.1 object prompts per label/chunk; 0 keeps YOLOE as support-only.")
    p.add_argument("--sam31_nontext_sparse_support", type=int, default=1,
                   help="Keep high-confidence YOLOE person masks as sparse support without extra SAM3.1 propagation (1/0).")
    p.add_argument("--sam31_max_text_prompt_objects", type=int, default=12,
                   help="Cap text-prompt objects collected per SAM3.1 query; keep slightly below the internal cap to leave room for tracker discoveries.")
    p.add_argument("--sam31_max_internal_objects", type=int, default=16,
                   help="SAM3.1 multiplex runtime object cap per query. Default 16 improves crowded scenes while staying under the 9GB target in Taylor probes.")

    # Detector selection
    p.add_argument("--detector", choices=["gdino", "yoloe"], default="gdino")

    # Grounding DINO
    p.add_argument("--gdino_config", default=None)
    p.add_argument("--gdino_checkpoint", default=None)

    # YOLOE
    p.add_argument("--yoloe_model", default="yoloe-11l-seg.pt",
                   help="YOLOE model name/path. Use 'openvision/yoloe26-l-seg' or 'yoloe26-l-seg' to try YOLOE-26 via Hugging Face.")
    p.add_argument("--yoloe_batch_size", type=int, default=4,
                   help="YOLOE discovery batch size.")
    p.add_argument("--yoloe_imgsz", type=int, default=0,
                   help="Optional YOLOE predict image size. 0 uses Ultralytics default; use 1280 with --detector_max_side 1280 for high-resolution discovery.")

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Discovery
    p.add_argument("--thing_prompts", default=None, help="Comma-separated.")
    p.add_argument("--stuff_prompts", default=None, help="Comma-separated.")
    p.add_argument("--disable_stuff_prompts", type=int, default=1,
                   help="Default 1 disables detector-side STUFF prompts in this THING-first frontend; set 0 to keep legacy sparse STUFF.")
    p.add_argument("--stuff_backend", default="efficientsam3", choices=["efficientsam3", "clipseg", "groupvit", "lseg"],
                   help="Per-frame STUFF backend used when --efficientsam3_stuff_enable=1.")
    p.add_argument("--efficientsam3_stuff_enable", type=int, default=0,
                   help="Run a per-frame STUFF pass. By default it is merged inside each chunk before global finalize.")
    p.add_argument("--efficientsam3_stuff_chunk_mode", type=int, default=1,
                   help="Run and merge STUFF inside each chunk before global finalize; required for pipeline-style Stage C outputs.")
    p.add_argument("--efficientsam3_stuff_prompts", default="wall surface,floor surface,ceiling surface",
                   help="Comma-separated STUFF labels queried by EfficientSAM3 when --efficientsam3_stuff_enable=1.")
    p.add_argument("--efficientsam3_stuff_stride", type=int, default=1,
                   help="Run EfficientSAM3 STUFF every N frames. Default 1 avoids frozen masks.")
    p.add_argument("--efficientsam3_stuff_confidence_threshold", type=float, default=0.01,
                   help="EfficientSAM3 text-prompt confidence threshold for STUFF masks.")
    p.add_argument("--efficientsam3_stuff_min_area_ratio", type=float, default=0.003,
                   help="Drop tiny EfficientSAM3 STUFF masks below this frame-area ratio.")
    p.add_argument("--efficientsam3_stuff_max_area_ratio", type=float, default=0.92,
                   help="Drop implausibly huge EfficientSAM3 STUFF masks above this frame-area ratio.")
    p.add_argument("--efficientsam3_stuff_max_masks_per_label", type=int, default=3,
                   help="Union at most this many EfficientSAM3 masks per STUFF label per frame.")
    p.add_argument("--efficientsam3_stuff_replace_existing", type=int, default=1,
                   help="Remove existing structure/STUFF tracks before appending EfficientSAM3 STUFF tracks.")
    p.add_argument("--efficientsam3_stuff_subtract_things", type=int, default=1,
                   help="Remove tracked THING pixels from EfficientSAM3 STUFF masks before saving/rendering.")
    p.add_argument("--efficientsam3_stuff_subtract_dilation", type=int, default=7,
                   help="Dilate tracked THING masks by this many pixels before subtracting them from STUFF masks.")
    p.add_argument("--efficientsam3_stuff_checkpoint", default=None,
                   help="Local EfficientSAM3 image checkpoint for STUFF. If omitted, auto-downloads a default model.")
    p.add_argument("--efficientsam3_stuff_repo_id", default=DEFAULT_EFFICIENTSAM3_REPO_ID,
                   help="Hugging Face repo used when auto-downloading EfficientSAM3 STUFF checkpoint.")
    p.add_argument("--efficientsam3_stuff_filename", default=DEFAULT_EFFICIENTSAM3_FILENAME,
                   help="Filename inside the HF repo used when auto-downloading EfficientSAM3 STUFF checkpoint.")
    p.add_argument("--efficientsam3_stuff_cache_dir", default="ckpts/EfficientSAM3",
                   help="Local cache dir for EfficientSAM3 STUFF downloads.")
    p.add_argument("--efficientsam3_stuff_backbone_type", default="efficientvit",
                   choices=["repvit", "tinyvit", "efficientvit"],
                   help="EfficientSAM3 STUFF student vision backbone family.")
    p.add_argument("--efficientsam3_stuff_model_name", default="b1",
                   help="EfficientSAM3 STUFF model variant, e.g. b0 / 11m / m1.1.")
    p.add_argument("--efficientsam3_stuff_text_encoder_type", default="MobileCLIP-S0",
                   help="Optional EfficientSAM3 LiteText encoder type, e.g. MobileCLIP-S1.")
    p.add_argument("--efficientsam3_stuff_text_context_length", type=int, default=16,
                   help="EfficientSAM3 STUFF text encoder context length.")
    p.add_argument("--efficientsam3_stuff_text_pos_embed_table_size", type=int, default=16,
                   help="Optional LiteText pos-embed table size; 0 uses the context length default.")
    p.add_argument("--efficientsam3_stuff_amp", type=int, default=1,
                   help="Use autocast during EfficientSAM3 STUFF inference on CUDA.")
    p.add_argument("--efficientsam3_stuff_worker", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--efficientsam3_stuff_frame_list", default=None, help=argparse.SUPPRESS)
    p.add_argument("--efficientsam3_stuff_output_pt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--clipseg_stuff_model", default="CIDAS/clipseg-rd64-refined",
                   help="Hugging Face CLIPSeg model id/path used when --stuff_backend=clipseg.")
    p.add_argument("--clipseg_stuff_prompts", default="wall,floor,ceiling",
                   help="Comma-separated open-vocabulary STUFF labels queried by CLIPSeg.")
    p.add_argument("--clipseg_stuff_confidence_threshold", type=float, default=0.50,
                   help="CLIPSeg probability threshold for per-pixel STUFF assignment.")
    p.add_argument("--clipseg_stuff_min_area_ratio", type=float, default=0.010,
                   help="Drop tiny CLIPSeg STUFF masks below this frame-area ratio.")
    p.add_argument("--clipseg_stuff_max_area_ratio", type=float, default=0.75,
                   help="Drop implausibly huge CLIPSeg STUFF masks above this frame-area ratio.")
    p.add_argument("--clipseg_stuff_morph_kernel", type=int, default=5,
                   help="Morphology kernel size for CLIPSeg STUFF cleanup; <=1 disables it.")
    p.add_argument("--clipseg_stuff_amp", type=int, default=1,
                   help="Use autocast during CLIPSeg STUFF inference on CUDA.")
    p.add_argument("--clipseg_stuff_batch_size", type=int, default=8,
                   help="Number of video frames batched per CLIPSeg STUFF worker step.")
    p.add_argument("--clipseg_stuff_inprocess", type=int, default=1,
                   help="Run CLIPSeg STUFF in-process and cache the model across chunks. Disable if VRAM gets tight.")
    p.add_argument("--clipseg_stuff_worker", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--clipseg_stuff_frame_list", default=None, help=argparse.SUPPRESS)
    p.add_argument("--clipseg_stuff_output_pt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--groupvit_stuff_model", default="nvidia/groupvit-gcc-yfcc",
                   help="Hugging Face GroupViT model id/path used when --stuff_backend=groupvit.")
    p.add_argument("--groupvit_stuff_prompts", default="wall,floor,ceiling",
                   help="Comma-separated open-vocabulary STUFF labels queried by GroupViT.")
    p.add_argument("--groupvit_stuff_background_prompts", default="other",
                   help="Comma-separated GroupViT reject/background labels. These labels compete in argmax but are not saved as STUFF tracks.")
    p.add_argument("--groupvit_stuff_confidence_threshold", type=float, default=0.34,
                   help="Minimum softmax probability for GroupViT per-pixel STUFF assignment.")
    p.add_argument("--groupvit_stuff_min_area_ratio", type=float, default=0.010,
                   help="Drop tiny GroupViT STUFF masks below this frame-area ratio.")
    p.add_argument("--groupvit_stuff_max_area_ratio", type=float, default=0.92,
                   help="Drop implausibly huge GroupViT STUFF masks above this frame-area ratio.")
    p.add_argument("--groupvit_stuff_morph_kernel", type=int, default=0,
                   help="Optional morphology kernel size for GroupViT STUFF cleanup; <=1 disables it.")
    p.add_argument("--groupvit_stuff_amp", type=int, default=1,
                   help="Use autocast during GroupViT STUFF inference on CUDA.")
    p.add_argument("--groupvit_stuff_batch_size", type=int, default=16,
                   help="Number of video frames batched per GroupViT STUFF worker step.")
    p.add_argument("--groupvit_stuff_worker", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--groupvit_stuff_frame_list", default=None, help=argparse.SUPPRESS)
    p.add_argument("--groupvit_stuff_output_pt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--lseg_stuff_repo_root", default="third_party/LSM",
                   help="LSM repo root used for its pure-PyTorch LSegFeatureExtractor wrapper.")
    p.add_argument("--lseg_stuff_checkpoint", default="ckpts/LSeg/demo_e200.ckpt",
                   help="LSeg demo_e200.ckpt checkpoint used by the LSM wrapper.")
    p.add_argument("--lseg_stuff_prompts", default="wall,floor,ceiling",
                   help="Comma-separated open-vocabulary STUFF labels queried by LSeg.")
    p.add_argument("--lseg_stuff_prompt_template", default="there is a {classname} in the scene",
                   help="Template sent to LSeg text encoder; {classname} is replaced with each label.")
    p.add_argument("--lseg_stuff_background_prompts",
                   default="person,people,clothing,hair,bag,chair,table,furniture,object,screen,sky,tree,vegetation,other",
                   help="Reject/background labels that compete in LSeg argmax but are not saved as STUFF tracks.")
    p.add_argument("--lseg_stuff_confidence_threshold", type=float, default=0.15,
                   help="Minimum softmax probability for LSeg per-pixel STUFF assignment.")
    p.add_argument("--lseg_stuff_min_area_ratio", type=float, default=0.010,
                   help="Drop tiny LSeg STUFF masks below this frame-area ratio.")
    p.add_argument("--lseg_stuff_max_area_ratio", type=float, default=0.85,
                   help="Drop implausibly huge LSeg STUFF masks above this frame-area ratio.")
    p.add_argument("--lseg_stuff_morph_kernel", type=int, default=3,
                   help="Small morphology kernel for LSeg STUFF cleanup; <=1 disables it.")
    p.add_argument("--lseg_stuff_batch_size", type=int, default=1,
                   help="Number of video frames batched per LSeg STUFF step.")
    p.add_argument("--lseg_stuff_max_side", type=int, default=512,
                   help="Resize longest frame side for LSeg STUFF. 0 uses chunk frame size.")
    p.add_argument("--lseg_stuff_half_res", type=int, default=0,
                   help="Use LSM half-res feature decode path (1/0). Default 0 returns full-resolution logits before final resize.")
    p.add_argument("--lseg_stuff_amp", type=int, default=1,
                   help="Use autocast during LSeg STUFF inference on CUDA.")
    p.add_argument("--lseg_stuff_device", default="auto",
                   help="Device for LSeg STUFF. 'auto' uses cuda:1 when multiple visible GPUs exist, else --device.")
    p.add_argument("--lseg_stuff_inprocess", type=int, default=1,
                   help="Run LSeg STUFF in-process and cache the model across chunks. Disable to isolate VRAM.")
    p.add_argument("--lseg_stuff_worker", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--lseg_stuff_frame_list", default=None, help=argparse.SUPPRESS)
    p.add_argument("--lseg_stuff_output_pt", default=None, help=argparse.SUPPRESS)
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--ann_frame_idx", type=int, default=0,
                   help="Base discovery frame index inside each chunk.")
    p.add_argument("--discovery_frame_stride", type=int, default=4,
                   help="Run detector every N frames inside each chunk so late-appearing objects can be seeded. Default 4 keeps Taylor-scale full-video runtime manageable; lower this for denser discovery.")
    p.add_argument("--max_thing_objects", type=int, default=14,
                   help="Max thing objects to track per chunk.")
    p.add_argument("--sam31_max_movable_objects", type=int, default=4,
                   help="For sam31_multiplex, max movable thing prompts tracked per chunk.")
    p.add_argument("--sam31_max_static_objects", type=int, default=1,
                   help="For sam31_multiplex, max static thing prompts tracked per chunk.")
    p.add_argument("--sam31_max_structure_objects", type=int, default=0,
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


def is_video(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTS


def extract_video_frames(
    video_path: str,
    out_dir: str,
    start: int,
    end: int,
    stride: int,
) -> List[str]:
    """Extract video frames as high-quality JPGs for Stage C.

    The previous runner imported Stage A's helper, which writes PNGs and pulls
    in geometry-model imports. JPG is enough for detection/video visualisation
    here and cuts avoidable disk I/O before SAM3.1 propagation starts.
    """
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_end = total - 1 if int(end) == -1 else min(int(end), total - 1)
    stride = max(int(stride), 1)

    paths: List[str] = []
    idx = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx > actual_end:
            break
        if idx >= int(start) and (idx - int(start)) % stride == 0:
            path = os.path.join(out_dir, f"frame_{count:06d}.jpg")
            cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            paths.append(path)
            count += 1
        idx += 1
    cap.release()
    print(f"Extracted {count} frames from {video_path}")
    return paths


def collect_image_paths(input_path: str, start: int, end: int, stride: int) -> Tuple[List[str], Optional[str]]:
    """Return frame paths plus an optional temporary directory."""
    if is_video(input_path):
        tmp = tempfile.mkdtemp(prefix="loger_frames_")
        return extract_video_frames(input_path, tmp, start, end, stride), tmp

    if os.path.isdir(input_path):
        paths = natsorted(
            glob.glob(os.path.join(input_path, "*.png"))
            + glob.glob(os.path.join(input_path, "*.jpg"))
            + glob.glob(os.path.join(input_path, "*.jpeg"))
        )
        paths = [p for p in paths if "depth" not in os.path.basename(p).lower()]
        end_idx = None if int(end) == -1 else int(end)
        return paths[int(start):end_idx:max(int(stride), 1)], None

    raise FileNotFoundError(
        f"Input path does not exist or is not a recognised format: {input_path}"
    )


def prepare_processing_image_paths(
    image_paths: List[str],
    max_side: int,
) -> Tuple[List[str], Optional[str], Tuple[int, int], Tuple[int, int]]:
    """Optionally downscale frames for inference while preserving output size."""
    if not image_paths:
        return image_paths, None, (0, 0), (0, 0)

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first frame: {image_paths[0]}")
    orig_h, orig_w = first.shape[:2]
    if int(max_side) <= 0 or max(orig_h, orig_w) <= int(max_side):
        return image_paths, None, (orig_h, orig_w), (orig_h, orig_w)

    scale = float(max_side) / float(max(orig_h, orig_w))
    proc_w = max(1, int(round(orig_w * scale)))
    proc_h = max(1, int(round(orig_h * scale)))
    tmp = tempfile.mkdtemp(prefix="loger_masklet_resized_")
    processed_paths: List[str] = []
    for idx, path in enumerate(image_paths):
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Failed to read frame: {path}")
        if frame.shape[0] != orig_h or frame.shape[1] != orig_w:
            local_scale = float(max_side) / float(max(frame.shape[:2]))
            local_w = max(1, int(round(frame.shape[1] * local_scale)))
            local_h = max(1, int(round(frame.shape[0] * local_scale)))
            resized = cv2.resize(frame, (local_w, local_h), interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(tmp, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        processed_paths.append(out_path)
    return processed_paths, tmp, (orig_h, orig_w), (proc_h, proc_w)


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
    if {a, b}.issubset({"person", "people"}):
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


def _mask_to_box_np(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )


def upscale_sparse_masklet_output(
    mo,
    target_h: int,
    target_w: int,
) -> SparseMaskletOutput:
    if not isinstance(mo, SparseMaskletOutput):
        raise TypeError("upscale_sparse_masklet_output expects SparseMaskletOutput")
    target_h = int(target_h)
    target_w = int(target_w)
    if mo.frame_height == target_h and mo.frame_width == target_w:
        return mo

    scaled_tracks: List[Dict] = []
    for track in mo.tracks:
        new_track = dict(track)
        new_track["mask_by_frame"] = {}
        new_track["box_by_frame"] = {}
        new_track["area_by_frame"] = {}
        new_track["q_by_frame"] = dict(track.get("q_by_frame", {}))
        new_track["frame_height"] = target_h
        new_track["frame_width"] = target_w
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            mask = _unpack_mask_np(packed, mo.frame_height, mo.frame_width)
            up_mask = cv2.resize(
                mask.astype(np.uint8),
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if not up_mask.any():
                continue
            new_track["mask_by_frame"][frame_idx] = _pack_mask_np(up_mask)
            new_track["box_by_frame"][frame_idx] = torch.from_numpy(_mask_to_box_np(up_mask))
            new_track["area_by_frame"][frame_idx] = (
                float(up_mask.sum()) / float(max(target_h * target_w, 1))
            )
        scaled_tracks.append(new_track)

    debug = dict(mo.debug)
    debug["processing_frame_size"] = (int(mo.frame_height), int(mo.frame_width))
    debug["output_frame_size"] = (target_h, target_w)
    return SparseMaskletOutput(
        tracks=scaled_tracks,
        num_masklets=mo.num_masklets,
        num_frames=mo.num_frames,
        frame_height=target_h,
        frame_width=target_w,
        debug=debug,
    )


def _clean_instance_mask_components(
    mask: np.ndarray,
    sem_group: int,
    label: str,
    *,
    image_area: int,
) -> np.ndarray:
    binary = mask.astype(bool)
    if int(sem_group) not in {SEMANTIC_GROUP_MOVABLE_THING, SEMANTIC_GROUP_STATIC_THING}:
        return binary

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8,
    )
    if num_labels <= 2:
        return binary

    comp_areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if comp_areas.size == 0:
        return binary

    order = np.argsort(comp_areas)[::-1]
    largest = float(comp_areas[order[0]])
    if largest <= 0.0:
        return binary

    label_l = str(label).strip().lower()
    is_person = _labels_compatible(label_l, "person") or _labels_compatible(label_l, "people")
    single_instance = int(sem_group) == SEMANTIC_GROUP_MOVABLE_THING
    if is_person:
        single_instance = True

    keep = np.zeros_like(binary, dtype=bool)
    max_components = 1 if single_instance else 3
    min_component_area = max(12.0, 0.00002 * float(max(image_area, 1)))
    if is_person:
        # Person masks can be split by instruments, occlusion, or detector/SAM
        # disagreement. Keep nearby body parts while still dropping far islands.
        max_components = 3
        min_component_area = max(min_component_area, 0.035 * largest)
    if not single_instance:
        min_component_area = max(min_component_area, 0.08 * largest)

    kept = 0
    main_rank = int(order[0])
    mx = float(stats[main_rank + 1, cv2.CC_STAT_LEFT])
    my = float(stats[main_rank + 1, cv2.CC_STAT_TOP])
    mw = float(stats[main_rank + 1, cv2.CC_STAT_WIDTH])
    mh = float(stats[main_rank + 1, cv2.CC_STAT_HEIGHT])
    main_box = (mx, my, mx + mw, my + mh)
    main_cx = mx + 0.5 * mw
    main_cy = my + 0.5 * mh

    for comp_rank in order:
        comp_area = float(comp_areas[comp_rank])
        if kept > 0 and comp_area < min_component_area:
            continue
        if is_person and kept > 0:
            x = float(stats[int(comp_rank) + 1, cv2.CC_STAT_LEFT])
            y = float(stats[int(comp_rank) + 1, cv2.CC_STAT_TOP])
            w = float(stats[int(comp_rank) + 1, cv2.CC_STAT_WIDTH])
            h = float(stats[int(comp_rank) + 1, cv2.CC_STAT_HEIGHT])
            cx = x + 0.5 * w
            cy = y + 0.5 * h

            expand_x = max(14.0, 0.45 * max(mw, 1.0))
            expand_y = max(18.0, 0.60 * max(mh, 1.0))
            expanded_main = (
                main_box[0] - expand_x,
                main_box[1] - expand_y,
                main_box[2] + expand_x,
                main_box[3] + expand_y,
            )
            overlaps_expanded = not (
                x + w < expanded_main[0]
                or x > expanded_main[2]
                or y + h < expanded_main[1]
                or y > expanded_main[3]
            )
            center_dx = abs(cx - main_cx) / max(0.5 * (mw + w), 1.0)
            center_dy = abs(cy - main_cy) / max(0.5 * (mh + h), 1.0)
            close_to_main = center_dx <= 1.25 and center_dy <= 1.35
            sizeable_component = comp_area >= 0.16 * largest
            if not (overlaps_expanded or close_to_main or sizeable_component):
                continue
        keep |= labels == int(comp_rank + 1)
        kept += 1
        if kept >= max_components:
            break

    return keep if keep.any() else binary


def _warp_mask_between_boxes(
    mask: np.ndarray,
    src_box: np.ndarray,
    dst_box: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    src = np.asarray(src_box, dtype=np.float32).reshape(-1)[:4]
    dst = np.asarray(dst_box, dtype=np.float32).reshape(-1)[:4]
    src_w = max(float(src[2] - src[0]), 1.0)
    src_h = max(float(src[3] - src[1]), 1.0)
    dst_w = max(float(dst[2] - dst[0]), 1.0)
    dst_h = max(float(dst[3] - dst[1]), 1.0)
    sx = float(np.clip(dst_w / src_w, 0.72, 1.38))
    sy = float(np.clip(dst_h / src_h, 0.72, 1.38))
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
    return (warped > 0).astype(np.uint8)


def _enforce_min_box_scale(
    box: np.ndarray,
    ref_box: np.ndarray,
    min_scale: float,
    H: int,
    W: int,
) -> np.ndarray:
    """Keep interpolated person boxes from collapsing after partial masks."""
    target = np.asarray(box, dtype=np.float32).reshape(-1)[:4].copy()
    ref = np.asarray(ref_box, dtype=np.float32).reshape(-1)[:4]
    cx = 0.5 * float(target[0] + target[2])
    cy = 0.5 * float(target[1] + target[3])
    tw = max(float(target[2] - target[0]), 1.0)
    th = max(float(target[3] - target[1]), 1.0)
    rw = max(float(ref[2] - ref[0]), 1.0)
    rh = max(float(ref[3] - ref[1]), 1.0)
    tw = max(tw, float(min_scale) * rw)
    th = max(th, float(min_scale) * rh)
    x1 = max(0.0, cx - 0.5 * tw)
    y1 = max(0.0, cy - 0.5 * th)
    x2 = min(float(W - 1), cx + 0.5 * tw)
    y2 = min(float(H - 1), cy + 0.5 * th)
    if x2 <= x1:
        x2 = min(float(W - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(H - 1), y1 + 1.0)
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


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


def _parse_comma_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _make_sparse_stuff_track(label: str, H: int, W: int) -> Dict:
    canonical = canonicalize_label(label)
    sem_group = label_to_group(canonical)
    return {
        "mask_by_frame": {},
        "box_by_frame": {},
        "q_by_frame": {},
        "area_by_frame": {},
        "L_sem": canonical,
        "G_sem": int(sem_group),
        "W_sem": float(DEFAULT_SEMANTIC_WEIGHTS.get(int(sem_group), 0.15)),
        "source_type": "stuff_static",
        "birth_frame": 0,
        "frame_height": int(H),
        "frame_width": int(W),
    }


def _subtract_thing_pixels_from_stuff_tracks(
    stuff_tracks: List[Dict],
    thing_tracks: List[Dict],
    H: int,
    W: int,
    dilation: int = 0,
) -> Tuple[List[Dict], Dict[str, int]]:
    if not stuff_tracks or not thing_tracks:
        return list(stuff_tracks), {
            "efficientsam3_stuff_subtract_thing_masks": 0,
            "efficientsam3_stuff_subtract_emptied_masks": 0,
            "efficientsam3_stuff_subtract_changed_masks": 0,
            "efficientsam3_stuff_subtract_dilation": int(max(dilation, 0)),
        }

    thing_union_by_frame: Dict[int, np.ndarray] = {}
    for track in thing_tracks:
        if str(track.get("source_type", "")) != "thing_tracked":
            continue
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            frame_idx = int(frame_idx)
            packed_np = np.asarray(packed, dtype=np.uint8)
            current = thing_union_by_frame.get(frame_idx)
            if current is None:
                thing_union_by_frame[frame_idx] = packed_np.copy()
            else:
                np.bitwise_or(current, packed_np, out=current)

    if not thing_union_by_frame:
        return list(stuff_tracks), {
            "efficientsam3_stuff_subtract_thing_masks": 0,
            "efficientsam3_stuff_subtract_emptied_masks": 0,
            "efficientsam3_stuff_subtract_changed_masks": 0,
            "efficientsam3_stuff_subtract_dilation": int(max(dilation, 0)),
        }

    cleaned_tracks: List[Dict] = []
    changed_masks = 0
    emptied_masks = 0
    touched_masks = 0
    dilation = int(max(dilation, 0))
    if dilation > 1:
        if dilation % 2 == 0:
            dilation += 1
        dilation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(dilation), int(dilation)),
        )
        dilated_union_by_frame: Dict[int, np.ndarray] = {}
    else:
        dilation_kernel = None
        dilated_union_by_frame = {}

    def _thing_union_for_subtract(frame_idx: int) -> Optional[np.ndarray]:
        packed_union = thing_union_by_frame.get(int(frame_idx))
        if packed_union is None:
            return None
        if dilation_kernel is None:
            return packed_union
        cached = dilated_union_by_frame.get(int(frame_idx))
        if cached is not None:
            return cached
        union_mask = _unpack_mask_np(packed_union, H, W).astype(np.uint8)
        dilated = cv2.dilate(union_mask, dilation_kernel, iterations=1).astype(bool)
        cached = _pack_mask_np(dilated)
        dilated_union_by_frame[int(frame_idx)] = cached
        return cached

    for track in stuff_tracks:
        mask_by_frame: Dict[int, np.ndarray] = {}
        box_by_frame: Dict[int, torch.Tensor] = {}
        q_by_frame: Dict[int, float] = {}
        area_by_frame: Dict[int, float] = {}

        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            frame_idx = int(frame_idx)
            packed_np = np.asarray(packed, dtype=np.uint8)
            thing_union = _thing_union_for_subtract(frame_idx)
            if thing_union is None:
                cleaned_packed = packed_np.copy()
            else:
                touched_masks += 1
                cleaned_packed = np.bitwise_and(packed_np, np.bitwise_not(thing_union))
                if not np.array_equal(cleaned_packed, packed_np):
                    changed_masks += 1

            mask = _unpack_mask_np(cleaned_packed, H, W)
            if not mask.any():
                emptied_masks += 1
                continue
            mask_by_frame[frame_idx] = _pack_mask_np(mask)
            box_by_frame[frame_idx] = torch.from_numpy(_mask_to_box_np(mask))
            q_by_frame[frame_idx] = float(track.get("q_by_frame", {}).get(frame_idx, 0.0))
            area_by_frame[frame_idx] = float(mask.sum()) / max(float(H * W), 1.0)

        if not mask_by_frame:
            continue
        cleaned = dict(track)
        cleaned["mask_by_frame"] = mask_by_frame
        cleaned["box_by_frame"] = box_by_frame
        cleaned["q_by_frame"] = q_by_frame
        cleaned["area_by_frame"] = area_by_frame
        cleaned["birth_frame"] = min(mask_by_frame)
        cleaned_tracks.append(cleaned)

    return cleaned_tracks, {
        "efficientsam3_stuff_subtract_thing_masks": int(touched_masks),
        "efficientsam3_stuff_subtract_emptied_masks": int(emptied_masks),
        "efficientsam3_stuff_subtract_changed_masks": int(changed_masks),
        "efficientsam3_stuff_subtract_dilation": int(dilation),
    }


def _merge_efficientsam3_stuff_tracks(
    mo: SparseMaskletOutput,
    payload: Dict,
    *,
    replace_existing: bool,
    subtract_things: bool,
    subtract_dilation: int = 0,
) -> SparseMaskletOutput:
    stuff_tracks = list(payload.get("tracks", []))
    if not stuff_tracks:
        debug = dict(mo.debug)
        debug["efficientsam3_stuff_tracks_added"] = 0
        debug["efficientsam3_stuff_masks_added"] = 0
        return SparseMaskletOutput(
            tracks=list(mo.tracks),
            num_masklets=len(mo.tracks),
            num_frames=mo.num_frames,
            frame_height=mo.frame_height,
            frame_width=mo.frame_width,
            debug=debug,
        )

    if int(payload.get("frame_height", -1)) != int(mo.frame_height) or int(payload.get("frame_width", -1)) != int(mo.frame_width):
        raise ValueError(
            "EfficientSAM3 STUFF frame size mismatch: "
            f"{payload.get('frame_height')}x{payload.get('frame_width')} vs "
            f"{mo.frame_height}x{mo.frame_width}"
        )
    if int(payload.get("num_frames", -1)) != int(mo.num_frames):
        raise ValueError(
            "EfficientSAM3 STUFF frame count mismatch: "
            f"{payload.get('num_frames')} vs {mo.num_frames}"
        )

    subtract_debug: Dict[str, int] = {}
    if subtract_things:
        thing_tracks = [
            track for track in mo.tracks
            if str(track.get("source_type", "")) == "thing_tracked"
        ]
        stuff_tracks, subtract_debug = _subtract_thing_pixels_from_stuff_tracks(
            stuff_tracks,
            thing_tracks,
            mo.frame_height,
            mo.frame_width,
            dilation=int(subtract_dilation),
        )

    kept_tracks = [
        track for track in mo.tracks
        if not (
            replace_existing
            and str(track.get("source_type", "")) in {"structure_tracked", "stuff_static"}
        )
    ]
    kept_tracks.extend(stuff_tracks)

    debug = dict(mo.debug)
    debug.update(payload.get("debug", {}))
    debug.update(subtract_debug)
    debug["efficientsam3_stuff_replaced_existing"] = int(len(mo.tracks) - len(kept_tracks) + len(stuff_tracks))
    debug["J_thing"] = sum(1 for track in kept_tracks if track["source_type"] == "thing_tracked")
    debug["J_structure"] = sum(1 for track in kept_tracks if track["source_type"] == "structure_tracked")
    debug["J_stuff"] = sum(1 for track in kept_tracks if track["source_type"] == "stuff_static")
    return SparseMaskletOutput(
        tracks=kept_tracks,
        num_masklets=len(kept_tracks),
        num_frames=mo.num_frames,
        frame_height=mo.frame_height,
        frame_width=mo.frame_width,
        debug=debug,
    )


def _resolve_efficientsam3_stuff_checkpoint(args: argparse.Namespace) -> str:
    if args.efficientsam3_stuff_checkpoint:
        checkpoint = os.path.expanduser(str(args.efficientsam3_stuff_checkpoint))
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"EfficientSAM3 STUFF checkpoint not found: {checkpoint}")
        return checkpoint

    from huggingface_hub import hf_hub_download

    cache_dir = os.path.expanduser(str(args.efficientsam3_stuff_cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=str(args.efficientsam3_stuff_repo_id),
        filename=str(args.efficientsam3_stuff_filename),
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )


def _run_efficientsam3_stuff_worker(args: argparse.Namespace) -> None:
    frame_list_path = args.efficientsam3_stuff_frame_list
    output_pt = args.efficientsam3_stuff_output_pt
    if not frame_list_path or not output_pt:
        raise ValueError("EfficientSAM3 STUFF worker requires frame list and output path.")

    with open(frame_list_path, "r", encoding="utf-8") as handle:
        image_paths = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
    if not image_paths:
        raise ValueError("EfficientSAM3 STUFF worker received no frames.")

    labels = _parse_comma_list(args.efficientsam3_stuff_prompts)
    if not labels:
        torch.save({"tracks": [], "debug": {"efficientsam3_stuff_skipped": "no_labels"}}, output_pt)
        return

    repo_root = os.path.dirname(os.path.abspath(__file__))
    efficient_root = os.path.join(repo_root, "third_party", "efficientsam3", "sam3")
    if efficient_root not in sys.path:
        sys.path.insert(0, efficient_root)

    from PIL import Image
    from sam3.model_builder import build_efficientsam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first EfficientSAM3 STUFF frame: {image_paths[0]}")
    H, W = first.shape[:2]

    checkpoint_path = _resolve_efficientsam3_stuff_checkpoint(args)
    pos_embed_table_size = (
        None
        if int(args.efficientsam3_stuff_text_pos_embed_table_size) <= 0
        else int(args.efficientsam3_stuff_text_pos_embed_table_size)
    )
    device = args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    print(
        "EfficientSAM3 STUFF worker loading "
        f"{checkpoint_path} labels={labels} frames={len(image_paths)} stride={args.efficientsam3_stuff_stride}",
        flush=True,
    )
    bpe_path = os.path.join(efficient_root, "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_efficientsam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
        backbone_type=str(args.efficientsam3_stuff_backbone_type),
        model_name=str(args.efficientsam3_stuff_model_name),
        text_encoder_type=args.efficientsam3_stuff_text_encoder_type,
        text_encoder_context_length=int(args.efficientsam3_stuff_text_context_length),
        text_encoder_pos_embed_table_size=pos_embed_table_size,
        interpolate_pos_embed=False,
        device=device,
    )
    stuff_builder_name = "efficientsam3_image"
    model.eval()
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(args.efficientsam3_stuff_confidence_threshold),
    )

    label_tracks = {
        canonicalize_label(label): _make_sparse_stuff_track(label, H, W)
        for label in labels
    }
    label_prompts = [(canonicalize_label(label), label) for label in labels]
    stride = max(int(args.efficientsam3_stuff_stride), 1)
    max_masks = max(int(args.efficientsam3_stuff_max_masks_per_label), 1)
    min_area_ratio = max(float(args.efficientsam3_stuff_min_area_ratio), 0.0)
    max_area_ratio = min(max(float(args.efficientsam3_stuff_max_area_ratio), min_area_ratio), 1.0)
    use_amp = bool(args.efficientsam3_stuff_amp) and str(device).startswith("cuda")
    total_masks = 0
    frames_with_any = 0
    quality_reject_counts: Dict[str, int] = {}
    t0 = time.time()

    with torch.inference_mode():
        for frame_idx, image_path in enumerate(image_paths):
            if frame_idx % stride != 0:
                continue
            image = Image.open(image_path).convert("RGB")
            frame_h, frame_w = image.height, image.width
            if frame_h != H or frame_w != W:
                raise ValueError(
                    f"EfficientSAM3 STUFF frame size changed at {image_path}: "
                    f"{frame_h}x{frame_w} vs {H}x{W}"
                )
            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                state = processor.set_image(image)
                frame_added = False
                for canonical, prompt in label_prompts:
                    processor.reset_all_prompts(state)
                    try:
                        state = processor.set_text_prompt(prompt=str(prompt), state=state)
                    except Exception as exc:
                        print(
                            f"EfficientSAM3 STUFF warning: prompt={prompt!r} "
                            f"frame={frame_idx} failed: {exc!r}",
                            flush=True,
                        )
                        continue
                    masks = state.get("masks")
                    scores = state.get("scores")
                    if masks is None or scores is None or int(masks.shape[0]) <= 0:
                        continue
                    order = torch.argsort(scores.detach().float().cpu(), descending=True).tolist()
                    union_mask = np.zeros((H, W), dtype=bool)
                    best_score = 0.0
                    kept = 0
                    for mask_idx in order:
                        if kept >= max_masks:
                            break
                        score = float(scores[int(mask_idx)].detach().float().cpu().item())
                        mask = masks[int(mask_idx)].detach().cpu().numpy()
                        mask = np.squeeze(mask).astype(bool)
                        if mask.shape != (H, W):
                            mask = cv2.resize(
                                mask.astype(np.uint8),
                                (W, H),
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(bool)
                        area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
                        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                            continue
                        keep, reason = passes_structure_mask_quality(
                            canonical,
                            mask,
                            _mask_to_box_np(mask),
                            score,
                            area_ratio,
                            H,
                            W,
                        )
                        if not keep:
                            quality_reject_counts[str(reason)] = (
                                int(quality_reject_counts.get(str(reason), 0)) + 1
                            )
                            continue
                        union_mask |= mask
                        best_score = max(best_score, score)
                        kept += 1
                    if kept <= 0 or not union_mask.any():
                        continue
                    area_ratio = float(union_mask.sum()) / max(float(H * W), 1.0)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    keep, reason = passes_structure_mask_quality(
                        canonical,
                        union_mask,
                        _mask_to_box_np(union_mask),
                        best_score,
                        area_ratio,
                        H,
                        W,
                    )
                    if not keep:
                        quality_reject_counts[str(reason)] = (
                            int(quality_reject_counts.get(str(reason), 0)) + 1
                        )
                        continue
                    track = label_tracks[canonical]
                    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(union_mask)
                    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(union_mask))
                    track["q_by_frame"][int(frame_idx)] = float(best_score)
                    track["area_by_frame"][int(frame_idx)] = area_ratio
                    if len(track["mask_by_frame"]) == 1:
                        track["birth_frame"] = int(frame_idx)
                    total_masks += 1
                    frame_added = True
                if frame_added:
                    frames_with_any += 1

            if (frame_idx + 1) % 100 == 0:
                print(
                    f"  EfficientSAM3 STUFF processed {frame_idx + 1}/{len(image_paths)} frames",
                    flush=True,
                )

    tracks = [track for track in label_tracks.values() if track["mask_by_frame"]]
    payload = {
        "format": "efficientsam3_stuff_tracks_v1",
        "num_frames": len(image_paths),
        "frame_height": H,
        "frame_width": W,
        "tracks": tracks,
        "debug": {
            "efficientsam3_stuff_checkpoint": checkpoint_path,
            "efficientsam3_stuff_builder": stuff_builder_name,
            "efficientsam3_stuff_labels": labels,
            "efficientsam3_stuff_stride": int(stride),
            "efficientsam3_stuff_tracks_added": int(len(tracks)),
            "efficientsam3_stuff_masks_added": int(total_masks),
            "efficientsam3_stuff_frames_with_any": int(frames_with_any),
            "efficientsam3_stuff_elapsed_seconds": float(time.time() - t0),
            "efficientsam3_stuff_quality_reject_counts": dict(quality_reject_counts),
        },
    }
    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(payload, output_pt)
    print(
        f"EfficientSAM3 STUFF worker saved {len(tracks)} tracks, "
        f"{total_masks} frame masks in {time.time() - t0:.1f}s -> {output_pt}",
        flush=True,
    )


def _run_efficientsam3_stuff_subprocess_payload(
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
    *,
    prefix: str = "loger_efficientsam3_stuff_",
) -> Dict:
    labels = _parse_comma_list(args.efficientsam3_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return {"tracks": [], "debug": {"efficientsam3_stuff_skipped": "disabled_or_no_labels"}}

    work_dir = tempfile.mkdtemp(prefix=prefix)
    temp_dirs.append(work_dir)
    frame_list_path = os.path.join(work_dir, "frames.txt")
    output_pt = os.path.join(work_dir, "stuff_tracks.pt")
    with open(frame_list_path, "w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"{path}\n")

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--efficientsam3_stuff_worker", "1",
        "--input", "__efficientsam3_stuff_worker__",
        "--efficientsam3_stuff_frame_list", frame_list_path,
        "--efficientsam3_stuff_output_pt", output_pt,
        "--device", str(args.device),
        "--efficientsam3_stuff_prompts", str(args.efficientsam3_stuff_prompts),
        "--efficientsam3_stuff_stride", str(args.efficientsam3_stuff_stride),
        "--efficientsam3_stuff_confidence_threshold", str(args.efficientsam3_stuff_confidence_threshold),
        "--efficientsam3_stuff_min_area_ratio", str(args.efficientsam3_stuff_min_area_ratio),
        "--efficientsam3_stuff_max_area_ratio", str(args.efficientsam3_stuff_max_area_ratio),
        "--efficientsam3_stuff_max_masks_per_label", str(args.efficientsam3_stuff_max_masks_per_label),
        "--efficientsam3_stuff_repo_id", str(args.efficientsam3_stuff_repo_id),
        "--efficientsam3_stuff_filename", str(args.efficientsam3_stuff_filename),
        "--efficientsam3_stuff_cache_dir", str(args.efficientsam3_stuff_cache_dir),
        "--efficientsam3_stuff_backbone_type", str(args.efficientsam3_stuff_backbone_type),
        "--efficientsam3_stuff_model_name", str(args.efficientsam3_stuff_model_name),
        "--efficientsam3_stuff_text_context_length", str(args.efficientsam3_stuff_text_context_length),
        "--efficientsam3_stuff_text_pos_embed_table_size", str(args.efficientsam3_stuff_text_pos_embed_table_size),
        "--efficientsam3_stuff_amp", str(args.efficientsam3_stuff_amp),
    ]
    if args.efficientsam3_stuff_checkpoint:
        cmd.extend(["--efficientsam3_stuff_checkpoint", str(args.efficientsam3_stuff_checkpoint)])
    if args.efficientsam3_stuff_text_encoder_type:
        cmd.extend(["--efficientsam3_stuff_text_encoder_type", str(args.efficientsam3_stuff_text_encoder_type)])

    print(
        "Running EfficientSAM3 STUFF subprocess "
        f"(labels={labels}, stride={args.efficientsam3_stuff_stride}) ...",
        flush=True,
    )
    t0 = time.time()
    subprocess.run(cmd, check=True)
    payload = torch.load(output_pt, map_location="cpu", weights_only=False)
    payload.setdefault("debug", {})["efficientsam3_stuff_subprocess_seconds"] = float(time.time() - t0)
    return payload


def _augment_with_efficientsam3_stuff_subprocess(
    mo: SparseMaskletOutput,
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
) -> SparseMaskletOutput:
    labels = _parse_comma_list(args.efficientsam3_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return mo

    t0 = time.time()
    payload = _run_efficientsam3_stuff_subprocess_payload(
        image_paths,
        args,
        temp_dirs,
    )
    augmented = _merge_efficientsam3_stuff_tracks(
        mo,
        payload,
        replace_existing=bool(args.efficientsam3_stuff_replace_existing),
        subtract_things=bool(args.efficientsam3_stuff_subtract_things),
        subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
    )
    print(
        f"EfficientSAM3 STUFF merge done in {time.time() - t0:.1f}s: "
        f"+{payload.get('debug', {}).get('efficientsam3_stuff_tracks_added', 0)} tracks, "
        f"+{payload.get('debug', {}).get('efficientsam3_stuff_masks_added', 0)} masks",
        flush=True,
    )
    return augmented


def _get_clipseg_stuff_model(args: argparse.Namespace):
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    device = args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model_id = str(args.clipseg_stuff_model)
    key = (model_id, str(device))
    cached = _CLIPSEG_STUFF_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1], device

    print(f"CLIPSeg STUFF in-process loading {model_id} on {device}", flush=True)
    processor = CLIPSegProcessor.from_pretrained(model_id, use_fast=True)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(device).eval()
    _CLIPSEG_STUFF_CACHE[key] = (processor, model)
    return processor, model, device


def _release_clipseg_stuff_cache() -> None:
    if not _CLIPSEG_STUFF_CACHE:
        return
    _CLIPSEG_STUFF_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _trim_cuda_cache() -> None:
    """Release inactive CUDA blocks between heavyweight model phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_clipseg_stuff_inprocess_payload(
    image_paths: List[str],
    args: argparse.Namespace,
) -> Dict:
    labels = _parse_comma_list(args.clipseg_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return {"tracks": [], "debug": {"clipseg_stuff_skipped": "disabled_or_no_labels"}}

    from PIL import Image

    if not image_paths:
        return {"tracks": [], "debug": {"clipseg_stuff_skipped": "no_frames"}}
    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first CLIPSeg STUFF frame: {image_paths[0]}")
    H, W = first.shape[:2]
    processor, model, device = _get_clipseg_stuff_model(args)
    model_id = str(args.clipseg_stuff_model)
    print(
        f"CLIPSeg STUFF in-process running {model_id} labels={labels} "
        f"frames={len(image_paths)} stride={args.efficientsam3_stuff_stride}",
        flush=True,
    )

    label_tracks = {
        canonicalize_label(label): _make_sparse_stuff_track(label, H, W)
        for label in labels
    }
    label_prompts = [(canonicalize_label(label), label) for label in labels]
    stride = max(int(args.efficientsam3_stuff_stride), 1)
    threshold = float(args.clipseg_stuff_confidence_threshold)
    min_area_ratio = max(float(args.clipseg_stuff_min_area_ratio), 0.0)
    max_area_ratio = min(max(float(args.clipseg_stuff_max_area_ratio), min_area_ratio), 1.0)
    kernel_size = max(int(args.clipseg_stuff_morph_kernel), 0)
    morph_kernel = (
        np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if kernel_size > 1
        else None
    )
    use_amp = bool(args.clipseg_stuff_amp) and str(device).startswith("cuda")
    batch_size = max(int(getattr(args, "clipseg_stuff_batch_size", 1)), 1)
    total_masks = 0
    frames_with_any = 0
    quality_reject_counts: Dict[str, int] = {}
    t0 = time.time()

    with torch.inference_mode():
        frame_jobs = [
            (int(frame_idx), image_path)
            for frame_idx, image_path in enumerate(image_paths)
            if int(frame_idx) % stride == 0
        ]
        for batch_start in range(0, len(frame_jobs), batch_size):
            batch_jobs = frame_jobs[batch_start:batch_start + batch_size]
            batch_images = []
            for _frame_idx, image_path in batch_jobs:
                image = Image.open(image_path).convert("RGB")
                frame_h, frame_w = image.height, image.width
                if frame_h != H or frame_w != W:
                    raise ValueError(
                        f"CLIPSeg STUFF frame size changed at {image_path}: "
                        f"{frame_h}x{frame_w} vs {H}x{W}"
                    )
                batch_images.append(image)

            prompts: List[str] = []
            prompt_images: List[Image.Image] = []
            for image in batch_images:
                for _canonical, prompt in label_prompts:
                    prompts.append(prompt)
                    prompt_images.append(image)

            amp_context = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                inputs = processor(
                    text=prompts,
                    images=prompt_images,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).detach().float().cpu().numpy()

            if probs.ndim == 2:
                probs = probs[None, ...]
            if probs.shape[0] != len(batch_jobs) * len(label_prompts):
                continue
            probs = probs.reshape(len(batch_jobs), len(label_prompts), probs.shape[-2], probs.shape[-1])

            for batch_idx, (frame_idx, _image_path) in enumerate(batch_jobs):
                resized_probs: List[np.ndarray] = []
                for prob in probs[batch_idx]:
                    resized_probs.append(
                        cv2.resize(
                            np.asarray(prob, dtype=np.float32),
                            (W, H),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    )
                if not resized_probs:
                    continue
                prob_stack = np.stack(resized_probs, axis=0)
                winner = np.argmax(prob_stack, axis=0)
                max_prob = np.max(prob_stack, axis=0)

                frame_added = False
                for label_idx, (canonical, _prompt) in enumerate(label_prompts):
                    mask = (winner == int(label_idx)) & (max_prob >= threshold)
                    if morph_kernel is not None:
                        mask_u8 = mask.astype(np.uint8)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, morph_kernel)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
                        mask = mask_u8.astype(bool)
                    if not mask.any():
                        continue
                    area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    score = float(np.percentile(prob_stack[label_idx][mask], 90))
                    keep, reason = passes_structure_mask_quality(
                        canonical,
                        mask,
                        _mask_to_box_np(mask),
                        score,
                        area_ratio,
                        H,
                        W,
                    )
                    if not keep:
                        quality_reject_counts[str(reason)] = (
                            int(quality_reject_counts.get(str(reason), 0)) + 1
                        )
                        continue

                    track = label_tracks[canonical]
                    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask)
                    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask))
                    track["q_by_frame"][int(frame_idx)] = float(score)
                    track["area_by_frame"][int(frame_idx)] = area_ratio
                    if len(track["mask_by_frame"]) == 1:
                        track["birth_frame"] = int(frame_idx)
                    total_masks += 1
                    frame_added = True
                if frame_added:
                    frames_with_any += 1

            last_frame_idx = int(batch_jobs[-1][0])
            if (last_frame_idx + 1) % 100 == 0 or batch_start + batch_size >= len(frame_jobs):
                print(
                    f"  CLIPSeg STUFF processed {last_frame_idx + 1}/{len(image_paths)} frames",
                    flush=True,
                )

    tracks = [track for track in label_tracks.values() if track["mask_by_frame"]]
    return {
        "format": "clipseg_stuff_tracks_v1",
        "num_frames": len(image_paths),
        "frame_height": H,
        "frame_width": W,
        "tracks": tracks,
        "debug": {
            "clipseg_stuff_model": model_id,
            "clipseg_stuff_labels": labels,
            "clipseg_stuff_stride": int(stride),
            "clipseg_stuff_batch_size": int(batch_size),
            "clipseg_stuff_threshold": float(threshold),
            "clipseg_stuff_tracks_added": int(len(tracks)),
            "clipseg_stuff_masks_added": int(total_masks),
            "clipseg_stuff_frames_with_any": int(frames_with_any),
            "clipseg_stuff_elapsed_seconds": float(time.time() - t0),
            "clipseg_stuff_inprocess_seconds": float(time.time() - t0),
            "clipseg_stuff_quality_reject_counts": dict(quality_reject_counts),
        },
    }


def _run_clipseg_stuff_worker(args: argparse.Namespace) -> None:
    frame_list_path = args.clipseg_stuff_frame_list
    output_pt = args.clipseg_stuff_output_pt
    if not frame_list_path or not output_pt:
        raise ValueError("CLIPSeg STUFF worker requires frame list and output path.")

    with open(frame_list_path, "r", encoding="utf-8") as handle:
        image_paths = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
    if not image_paths:
        raise ValueError("CLIPSeg STUFF worker received no frames.")

    labels = _parse_comma_list(args.clipseg_stuff_prompts)
    if not labels:
        torch.save({"tracks": [], "debug": {"clipseg_stuff_skipped": "no_labels"}}, output_pt)
        return

    from PIL import Image
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first CLIPSeg STUFF frame: {image_paths[0]}")
    H, W = first.shape[:2]
    device = args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model_id = str(args.clipseg_stuff_model)
    print(
        f"CLIPSeg STUFF worker loading {model_id} labels={labels} "
        f"frames={len(image_paths)} stride={args.efficientsam3_stuff_stride}",
        flush=True,
    )
    processor = CLIPSegProcessor.from_pretrained(model_id, use_fast=True)
    model = CLIPSegForImageSegmentation.from_pretrained(model_id).to(device).eval()

    label_tracks = {
        canonicalize_label(label): _make_sparse_stuff_track(label, H, W)
        for label in labels
    }
    label_prompts = [(canonicalize_label(label), label) for label in labels]
    stride = max(int(args.efficientsam3_stuff_stride), 1)
    threshold = float(args.clipseg_stuff_confidence_threshold)
    min_area_ratio = max(float(args.clipseg_stuff_min_area_ratio), 0.0)
    max_area_ratio = min(max(float(args.clipseg_stuff_max_area_ratio), min_area_ratio), 1.0)
    kernel_size = max(int(args.clipseg_stuff_morph_kernel), 0)
    morph_kernel = (
        np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if kernel_size > 1
        else None
    )
    use_amp = bool(args.clipseg_stuff_amp) and str(device).startswith("cuda")
    batch_size = max(int(getattr(args, "clipseg_stuff_batch_size", 1)), 1)
    total_masks = 0
    frames_with_any = 0
    quality_reject_counts: Dict[str, int] = {}
    t0 = time.time()

    with torch.inference_mode():
        frame_jobs = [
            (int(frame_idx), image_path)
            for frame_idx, image_path in enumerate(image_paths)
            if int(frame_idx) % stride == 0
        ]
        for batch_start in range(0, len(frame_jobs), batch_size):
            batch_jobs = frame_jobs[batch_start:batch_start + batch_size]
            batch_images = []
            for _frame_idx, image_path in batch_jobs:
                image = Image.open(image_path).convert("RGB")
                frame_h, frame_w = image.height, image.width
                if frame_h != H or frame_w != W:
                    raise ValueError(
                        f"CLIPSeg STUFF frame size changed at {image_path}: "
                        f"{frame_h}x{frame_w} vs {H}x{W}"
                    )
                batch_images.append(image)

            prompts: List[str] = []
            prompt_images: List[Image.Image] = []
            for image in batch_images:
                for _canonical, prompt in label_prompts:
                    prompts.append(prompt)
                    prompt_images.append(image)

            amp_context = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                inputs = processor(
                    text=prompts,
                    images=prompt_images,
                    return_tensors="pt",
                ).to(device)
                logits = model(**inputs).logits
                probs = torch.sigmoid(logits).detach().float().cpu().numpy()

            if probs.ndim == 2:
                probs = probs[None, ...]
            if probs.shape[0] != len(batch_jobs) * len(label_prompts):
                continue
            probs = probs.reshape(len(batch_jobs), len(label_prompts), probs.shape[-2], probs.shape[-1])

            for batch_idx, (frame_idx, _image_path) in enumerate(batch_jobs):
                resized_probs: List[np.ndarray] = []
                for prob in probs[batch_idx]:
                    resized_probs.append(
                        cv2.resize(
                            np.asarray(prob, dtype=np.float32),
                            (W, H),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    )
                if not resized_probs:
                    continue
                prob_stack = np.stack(resized_probs, axis=0)
                winner = np.argmax(prob_stack, axis=0)
                max_prob = np.max(prob_stack, axis=0)

                frame_added = False
                for label_idx, (canonical, _prompt) in enumerate(label_prompts):
                    mask = (winner == int(label_idx)) & (max_prob >= threshold)
                    if morph_kernel is not None:
                        mask_u8 = mask.astype(np.uint8)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, morph_kernel)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
                        mask = mask_u8.astype(bool)
                    if not mask.any():
                        continue
                    area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    score = float(np.percentile(prob_stack[label_idx][mask], 90))
                    keep, reason = passes_structure_mask_quality(
                        canonical,
                        mask,
                        _mask_to_box_np(mask),
                        score,
                        area_ratio,
                        H,
                        W,
                    )
                    if not keep:
                        quality_reject_counts[str(reason)] = (
                            int(quality_reject_counts.get(str(reason), 0)) + 1
                        )
                        continue

                    track = label_tracks[canonical]
                    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask)
                    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask))
                    track["q_by_frame"][int(frame_idx)] = float(score)
                    track["area_by_frame"][int(frame_idx)] = area_ratio
                    if len(track["mask_by_frame"]) == 1:
                        track["birth_frame"] = int(frame_idx)
                    total_masks += 1
                    frame_added = True
                if frame_added:
                    frames_with_any += 1

            last_frame_idx = int(batch_jobs[-1][0])
            if (last_frame_idx + 1) % 100 == 0 or batch_start + batch_size >= len(frame_jobs):
                print(
                    f"  CLIPSeg STUFF processed {last_frame_idx + 1}/{len(image_paths)} frames",
                    flush=True,
                )

    tracks = [track for track in label_tracks.values() if track["mask_by_frame"]]
    payload = {
        "format": "clipseg_stuff_tracks_v1",
        "num_frames": len(image_paths),
        "frame_height": H,
        "frame_width": W,
        "tracks": tracks,
        "debug": {
            "clipseg_stuff_model": model_id,
            "clipseg_stuff_labels": labels,
            "clipseg_stuff_stride": int(stride),
            "clipseg_stuff_batch_size": int(batch_size),
            "clipseg_stuff_threshold": float(threshold),
            "clipseg_stuff_tracks_added": int(len(tracks)),
            "clipseg_stuff_masks_added": int(total_masks),
            "clipseg_stuff_frames_with_any": int(frames_with_any),
            "clipseg_stuff_elapsed_seconds": float(time.time() - t0),
            "clipseg_stuff_quality_reject_counts": dict(quality_reject_counts),
        },
    }
    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(payload, output_pt)
    print(
        f"CLIPSeg STUFF worker saved {len(tracks)} tracks, "
        f"{total_masks} frame masks in {time.time() - t0:.1f}s -> {output_pt}",
        flush=True,
    )


def _run_clipseg_stuff_subprocess_payload(
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
    *,
    prefix: str = "loger_clipseg_stuff_",
) -> Dict:
    labels = _parse_comma_list(args.clipseg_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return {"tracks": [], "debug": {"clipseg_stuff_skipped": "disabled_or_no_labels"}}

    work_dir = tempfile.mkdtemp(prefix=prefix)
    temp_dirs.append(work_dir)
    frame_list_path = os.path.join(work_dir, "frames.txt")
    output_pt = os.path.join(work_dir, "stuff_tracks.pt")
    with open(frame_list_path, "w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"{path}\n")

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--clipseg_stuff_worker", "1",
        "--input", "__clipseg_stuff_worker__",
        "--clipseg_stuff_frame_list", frame_list_path,
        "--clipseg_stuff_output_pt", output_pt,
        "--device", str(args.device),
        "--efficientsam3_stuff_stride", str(args.efficientsam3_stuff_stride),
        "--clipseg_stuff_model", str(args.clipseg_stuff_model),
        "--clipseg_stuff_prompts", str(args.clipseg_stuff_prompts),
        "--clipseg_stuff_confidence_threshold", str(args.clipseg_stuff_confidence_threshold),
        "--clipseg_stuff_min_area_ratio", str(args.clipseg_stuff_min_area_ratio),
        "--clipseg_stuff_max_area_ratio", str(args.clipseg_stuff_max_area_ratio),
        "--clipseg_stuff_morph_kernel", str(args.clipseg_stuff_morph_kernel),
        "--clipseg_stuff_amp", str(args.clipseg_stuff_amp),
        "--clipseg_stuff_batch_size", str(args.clipseg_stuff_batch_size),
    ]

    print(
        "Running CLIPSeg STUFF subprocess "
        f"(labels={labels}, stride={args.efficientsam3_stuff_stride}, "
        f"batch={args.clipseg_stuff_batch_size}) ...",
        flush=True,
    )
    t0 = time.time()
    subprocess.run(cmd, check=True)
    payload = torch.load(output_pt, map_location="cpu", weights_only=False)
    payload.setdefault("debug", {})["clipseg_stuff_subprocess_seconds"] = float(time.time() - t0)
    return payload


def _augment_with_clipseg_stuff_subprocess(
    mo: SparseMaskletOutput,
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
) -> SparseMaskletOutput:
    labels = _parse_comma_list(args.clipseg_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return mo

    t0 = time.time()
    payload = _run_clipseg_stuff_subprocess_payload(
        image_paths,
        args,
        temp_dirs,
    )
    augmented = _merge_efficientsam3_stuff_tracks(
        mo,
        payload,
        replace_existing=bool(args.efficientsam3_stuff_replace_existing),
        subtract_things=bool(args.efficientsam3_stuff_subtract_things),
        subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
    )
    debug = payload.get("debug", {})
    print(
        f"CLIPSeg STUFF merge done in {time.time() - t0:.1f}s: "
        f"+{debug.get('clipseg_stuff_tracks_added', 0)} tracks, "
        f"+{debug.get('clipseg_stuff_masks_added', 0)} masks",
        flush=True,
    )
    return augmented


def _run_groupvit_stuff_worker(args: argparse.Namespace) -> None:
    frame_list_path = args.groupvit_stuff_frame_list
    output_pt = args.groupvit_stuff_output_pt
    if not frame_list_path or not output_pt:
        raise ValueError("GroupViT STUFF worker requires frame list and output path.")

    with open(frame_list_path, "r", encoding="utf-8") as handle:
        image_paths = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
    if not image_paths:
        raise ValueError("GroupViT STUFF worker received no frames.")

    labels = _parse_comma_list(args.groupvit_stuff_prompts)
    if not labels:
        torch.save({"tracks": [], "debug": {"groupvit_stuff_skipped": "no_labels"}}, output_pt)
        return

    from PIL import Image
    from transformers import AutoProcessor, GroupViTModel

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first GroupViT STUFF frame: {image_paths[0]}")
    H, W = first.shape[:2]
    device = args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    model_id = str(args.groupvit_stuff_model)
    background_labels = _parse_comma_list(args.groupvit_stuff_background_prompts)
    all_prompts = list(labels) + list(background_labels)
    print(
        f"GroupViT STUFF worker loading {model_id} labels={labels} "
        f"background={background_labels} frames={len(image_paths)} "
        f"stride={args.efficientsam3_stuff_stride}",
        flush=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model = GroupViTModel.from_pretrained(model_id).to(device).eval()

    label_tracks = {
        canonicalize_label(label): _make_sparse_stuff_track(label, H, W)
        for label in labels
    }
    label_prompts = [(canonicalize_label(label), label) for label in labels]
    stride = max(int(args.efficientsam3_stuff_stride), 1)
    threshold = float(args.groupvit_stuff_confidence_threshold)
    min_area_ratio = max(float(args.groupvit_stuff_min_area_ratio), 0.0)
    max_area_ratio = min(max(float(args.groupvit_stuff_max_area_ratio), min_area_ratio), 1.0)
    kernel_size = max(int(args.groupvit_stuff_morph_kernel), 0)
    morph_kernel = (
        np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if kernel_size > 1
        else None
    )
    use_amp = bool(args.groupvit_stuff_amp) and str(device).startswith("cuda")
    batch_size = max(int(getattr(args, "groupvit_stuff_batch_size", 1)), 1)
    total_masks = 0
    frames_with_any = 0
    quality_reject_counts: Dict[str, int] = {}
    t0 = time.time()

    with torch.inference_mode():
        frame_jobs = [
            (int(frame_idx), image_path)
            for frame_idx, image_path in enumerate(image_paths)
            if int(frame_idx) % stride == 0
        ]
        for batch_start in range(0, len(frame_jobs), batch_size):
            batch_jobs = frame_jobs[batch_start:batch_start + batch_size]
            batch_images = []
            for _frame_idx, image_path in batch_jobs:
                image = Image.open(image_path).convert("RGB")
                frame_h, frame_w = image.height, image.width
                if frame_h != H or frame_w != W:
                    raise ValueError(
                        f"GroupViT STUFF frame size changed at {image_path}: "
                        f"{frame_h}x{frame_w} vs {H}x{W}"
                    )
                batch_images.append(image)

            amp_context = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                inputs = processor(
                    text=all_prompts,
                    images=batch_images,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs, output_segmentation=True)
                logits = outputs.segmentation_logits
                if logits is None:
                    raise RuntimeError("GroupViT did not return segmentation_logits.")
                probs = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()

            if probs.ndim != 4 or probs.shape[0] != len(batch_jobs):
                raise RuntimeError(f"Unexpected GroupViT probability shape: {probs.shape}")
            if probs.shape[1] < len(label_prompts):
                raise RuntimeError(
                    f"GroupViT returned {probs.shape[1]} labels for {len(label_prompts)} prompts."
                )

            for batch_idx, (frame_idx, _image_path) in enumerate(batch_jobs):
                resized_probs: List[np.ndarray] = []
                for prob in probs[batch_idx]:
                    resized_probs.append(
                        cv2.resize(
                            np.asarray(prob, dtype=np.float32),
                            (W, H),
                            interpolation=cv2.INTER_LINEAR,
                        )
                    )
                if not resized_probs:
                    continue
                prob_stack = np.stack(resized_probs, axis=0)
                winner = np.argmax(prob_stack, axis=0)
                max_prob = np.max(prob_stack, axis=0)

                frame_added = False
                for label_idx, (canonical, _prompt) in enumerate(label_prompts):
                    mask = (winner == int(label_idx)) & (max_prob >= threshold)
                    if morph_kernel is not None:
                        mask_u8 = mask.astype(np.uint8)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, morph_kernel)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
                        mask = mask_u8.astype(bool)
                    if not mask.any():
                        continue
                    area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    score = float(np.percentile(prob_stack[label_idx][mask], 90))
                    keep, reason = passes_structure_mask_quality(
                        canonical,
                        mask,
                        _mask_to_box_np(mask),
                        score,
                        area_ratio,
                        H,
                        W,
                    )
                    if not keep:
                        quality_reject_counts[str(reason)] = (
                            int(quality_reject_counts.get(str(reason), 0)) + 1
                        )
                        continue

                    track = label_tracks[canonical]
                    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask)
                    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask))
                    track["q_by_frame"][int(frame_idx)] = float(score)
                    track["area_by_frame"][int(frame_idx)] = area_ratio
                    if len(track["mask_by_frame"]) == 1:
                        track["birth_frame"] = int(frame_idx)
                    total_masks += 1
                    frame_added = True
                if frame_added:
                    frames_with_any += 1

            last_frame_idx = int(batch_jobs[-1][0])
            if (last_frame_idx + 1) % 100 == 0 or batch_start + batch_size >= len(frame_jobs):
                print(
                    f"  GroupViT STUFF processed {last_frame_idx + 1}/{len(image_paths)} frames",
                    flush=True,
                )

    tracks = [track for track in label_tracks.values() if track["mask_by_frame"]]
    payload = {
        "format": "groupvit_stuff_tracks_v1",
        "num_frames": len(image_paths),
        "frame_height": H,
        "frame_width": W,
        "tracks": tracks,
        "debug": {
            "groupvit_stuff_model": model_id,
            "groupvit_stuff_labels": labels,
            "groupvit_stuff_background_labels": background_labels,
            "groupvit_stuff_stride": int(stride),
            "groupvit_stuff_batch_size": int(batch_size),
            "groupvit_stuff_threshold": float(threshold),
            "groupvit_stuff_tracks_added": int(len(tracks)),
            "groupvit_stuff_masks_added": int(total_masks),
            "groupvit_stuff_frames_with_any": int(frames_with_any),
            "groupvit_stuff_elapsed_seconds": float(time.time() - t0),
            "groupvit_stuff_quality_reject_counts": dict(quality_reject_counts),
        },
    }
    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(payload, output_pt)
    print(
        f"GroupViT STUFF worker saved {len(tracks)} tracks, "
        f"{total_masks} frame masks in {time.time() - t0:.1f}s -> {output_pt}",
        flush=True,
    )


def _run_groupvit_stuff_subprocess_payload(
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
    *,
    prefix: str = "loger_groupvit_stuff_",
) -> Dict:
    labels = _parse_comma_list(args.groupvit_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return {"tracks": [], "debug": {"groupvit_stuff_skipped": "disabled_or_no_labels"}}

    work_dir = tempfile.mkdtemp(prefix=prefix)
    temp_dirs.append(work_dir)
    frame_list_path = os.path.join(work_dir, "frames.txt")
    output_pt = os.path.join(work_dir, "stuff_tracks.pt")
    with open(frame_list_path, "w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"{path}\n")

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--groupvit_stuff_worker", "1",
        "--input", "__groupvit_stuff_worker__",
        "--groupvit_stuff_frame_list", frame_list_path,
        "--groupvit_stuff_output_pt", output_pt,
        "--device", str(args.device),
        "--efficientsam3_stuff_stride", str(args.efficientsam3_stuff_stride),
        "--groupvit_stuff_model", str(args.groupvit_stuff_model),
        "--groupvit_stuff_prompts", str(args.groupvit_stuff_prompts),
        "--groupvit_stuff_background_prompts", str(args.groupvit_stuff_background_prompts),
        "--groupvit_stuff_confidence_threshold", str(args.groupvit_stuff_confidence_threshold),
        "--groupvit_stuff_min_area_ratio", str(args.groupvit_stuff_min_area_ratio),
        "--groupvit_stuff_max_area_ratio", str(args.groupvit_stuff_max_area_ratio),
        "--groupvit_stuff_morph_kernel", str(args.groupvit_stuff_morph_kernel),
        "--groupvit_stuff_amp", str(args.groupvit_stuff_amp),
        "--groupvit_stuff_batch_size", str(args.groupvit_stuff_batch_size),
    ]

    print(
        "Running GroupViT STUFF subprocess "
        f"(labels={labels}, stride={args.efficientsam3_stuff_stride}, "
        f"batch={args.groupvit_stuff_batch_size}) ...",
        flush=True,
    )
    t0 = time.time()
    subprocess.run(cmd, check=True)
    payload = torch.load(output_pt, map_location="cpu", weights_only=False)
    payload.setdefault("debug", {})["groupvit_stuff_subprocess_seconds"] = float(time.time() - t0)
    return payload


def _augment_with_groupvit_stuff_subprocess(
    mo: SparseMaskletOutput,
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
) -> SparseMaskletOutput:
    labels = _parse_comma_list(args.groupvit_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return mo

    t0 = time.time()
    payload = _run_groupvit_stuff_subprocess_payload(
        image_paths,
        args,
        temp_dirs,
    )
    augmented = _merge_efficientsam3_stuff_tracks(
        mo,
        payload,
        replace_existing=bool(args.efficientsam3_stuff_replace_existing),
        subtract_things=bool(args.efficientsam3_stuff_subtract_things),
        subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
    )
    debug = payload.get("debug", {})
    print(
        f"GroupViT STUFF merge done in {time.time() - t0:.1f}s: "
        f"+{debug.get('groupvit_stuff_tracks_added', 0)} tracks, "
        f"+{debug.get('groupvit_stuff_masks_added', 0)} masks",
        flush=True,
    )
    return augmented


def _resolve_lseg_stuff_device(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "lseg_stuff_device", "auto")).strip()
    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return "cuda:1"
    return args.device if str(args.device).startswith("cuda") and torch.cuda.is_available() else "cpu"


def _resolve_lseg_repo_root(args: argparse.Namespace) -> str:
    repo_root = os.path.abspath(os.path.expanduser(str(args.lseg_stuff_repo_root)))
    if os.path.isdir(os.path.join(repo_root, "large_spatial_model")):
        return repo_root
    sibling_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LSM")
    )
    if os.path.isdir(os.path.join(sibling_root, "large_spatial_model")):
        print(
            f"LSeg STUFF repo root not found at {repo_root}; using {sibling_root}",
            flush=True,
        )
        return sibling_root
    # This fallback is only for interactive experiments where we cloned LSM
    # outside the project tree to avoid vendoring a large third-party repo.
    probe_root = "/tmp/lsm_probe"
    if os.path.isdir(os.path.join(probe_root, "large_spatial_model")):
        print(
            f"LSeg STUFF repo root not found at {repo_root}; using {probe_root}",
            flush=True,
        )
        return probe_root
    raise FileNotFoundError(
        "LSeg STUFF requires the LSM repo for its pure-PyTorch LSeg wrapper. "
        f"Expected {repo_root}; clone https://github.com/NVlabs/LSM.git with submodules "
        "or pass --lseg_stuff_repo_root."
    )


def _resolve_lseg_checkpoint(args: argparse.Namespace) -> str:
    checkpoint = os.path.abspath(os.path.expanduser(str(args.lseg_stuff_checkpoint)))
    if os.path.exists(checkpoint):
        return checkpoint
    raise FileNotFoundError(
        f"LSeg checkpoint not found: {checkpoint}. "
        "Download LSeg demo_e200.ckpt, e.g. the LSM README Google Drive id "
        "1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7, and pass --lseg_stuff_checkpoint."
    )


def _resolve_lseg_load_checkpoint(checkpoint: str) -> str:
    base, ext = os.path.splitext(checkpoint)
    if ext.lower() == ".ckpt":
        converted = f"{base}_net_state.pt"
        if os.path.exists(converted):
            return converted
    return checkpoint


def _get_lseg_stuff_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, str, str]:
    repo_root = _resolve_lseg_repo_root(args)
    checkpoint = _resolve_lseg_checkpoint(args)
    load_checkpoint = _resolve_lseg_load_checkpoint(checkpoint)
    device = _resolve_lseg_stuff_device(args)
    half_res = bool(int(getattr(args, "lseg_stuff_half_res", 0)))
    key = (repo_root, load_checkpoint, str(device), half_res)
    cached = _LSEG_STUFF_CACHE.get(key)
    if cached is not None:
        return cached, device, repo_root

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from large_spatial_model.lseg import LSegFeatureExtractor

    print(
        f"LSeg STUFF loading via LSM wrapper repo={repo_root} "
        f"checkpoint={load_checkpoint} device={device}",
        flush=True,
    )
    prev_device: Optional[int] = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        prev_device = torch.cuda.current_device()
        torch.cuda.set_device(torch.device(device))
    try:
        ckpt = torch.load(load_checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {
            k[len("net."):]: v
            for k, v in state_dict.items()
            if str(k).startswith("net.")
        }
        if not state_dict and isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
            state_dict = dict(ckpt["state_dict"])
        if not state_dict:
            raise RuntimeError(f"LSeg checkpoint has no net.* weights: {load_checkpoint}")
        model = LSegFeatureExtractor(half_res=half_res)
        model.load_state_dict(state_dict, strict=True)
        del ckpt
        del state_dict
    finally:
        if prev_device is not None:
            torch.cuda.set_device(prev_device)
    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    _LSEG_STUFF_CACHE[key] = model
    return model, device, repo_root


def _release_lseg_stuff_cache() -> None:
    if not _LSEG_STUFF_CACHE:
        return
    _LSEG_STUFF_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_lseg_rgb_tensor(
    image_path: str,
    *,
    max_side: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read LSeg STUFF frame: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    if int(max_side) > 0 and max(H, W) > int(max_side):
        scale = float(max_side) / float(max(H, W))
        new_w = max(16, int(round((W * scale) / 16.0)) * 16)
        new_h = max(16, int(round((H * scale) / 16.0)) * 16)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor, (H, W)


def _format_lseg_text_prompt(label: str, template: str) -> str:
    label = str(label).strip()
    template = str(template).strip()
    if not template:
        return label
    if "{classname}" in template:
        return template.replace("{classname}", label)
    return f"{template} {label}".strip()


def _lseg_payload_from_probs(
    probs: np.ndarray,
    label_prompts: List[Tuple[str, str]],
    image_paths: List[str],
    *,
    H: int,
    W: int,
    threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
    morph_kernel: Optional[np.ndarray],
) -> Tuple[List[Dict], Dict[str, int]]:
    label_tracks = {
        canonicalize_label(label): _make_sparse_stuff_track(label, H, W)
        for _canonical, label in label_prompts
    }
    total_masks = 0
    frames_with_any = 0
    for frame_idx in range(probs.shape[0]):
        prob_stack = probs[frame_idx]
        winner = np.argmax(prob_stack, axis=0)
        max_prob = np.max(prob_stack, axis=0)
        frame_added = False
        for label_idx, (canonical, _prompt) in enumerate(label_prompts):
            mask = (winner == int(label_idx)) & (max_prob >= threshold)
            if morph_kernel is not None:
                mask_u8 = mask.astype(np.uint8)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, morph_kernel)
                mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, morph_kernel)
                mask = mask_u8.astype(bool)
            if not mask.any():
                continue
            area_ratio = float(mask.sum()) / max(float(H * W), 1.0)
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            score = float(np.percentile(prob_stack[label_idx][mask], 90))
            track = label_tracks[canonical]
            track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask)
            track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask))
            track["q_by_frame"][int(frame_idx)] = float(score)
            track["area_by_frame"][int(frame_idx)] = area_ratio
            if len(track["mask_by_frame"]) == 1:
                track["birth_frame"] = int(frame_idx)
            total_masks += 1
            frame_added = True
        if frame_added:
            frames_with_any += 1
    tracks = [track for track in label_tracks.values() if track["mask_by_frame"]]
    return tracks, {
        "lseg_stuff_tracks_added": int(len(tracks)),
        "lseg_stuff_masks_added": int(total_masks),
        "lseg_stuff_frames_with_any": int(frames_with_any),
        "lseg_stuff_num_input_frames": int(len(image_paths)),
    }


def _run_lseg_stuff_inprocess_payload(
    image_paths: List[str],
    args: argparse.Namespace,
) -> Dict:
    labels = _parse_comma_list(args.lseg_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return {"tracks": [], "debug": {"lseg_stuff_skipped": "disabled_or_no_labels"}}
    if not image_paths:
        return {"tracks": [], "debug": {"lseg_stuff_skipped": "no_frames"}}

    first = cv2.imread(image_paths[0], cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(f"Failed to read first LSeg STUFF frame: {image_paths[0]}")
    H, W = first.shape[:2]
    model, device, repo_root = _get_lseg_stuff_model(args)
    background_labels = _parse_comma_list(args.lseg_stuff_background_prompts)
    prompt_template = str(args.lseg_stuff_prompt_template)
    all_prompt_labels = list(labels) + list(background_labels)
    all_prompts = [
        _format_lseg_text_prompt(label, prompt_template)
        for label in all_prompt_labels
    ]
    label_prompts = [(canonicalize_label(label), label) for label in labels]
    stride = max(int(args.efficientsam3_stuff_stride), 1)
    threshold = float(args.lseg_stuff_confidence_threshold)
    min_area_ratio = max(float(args.lseg_stuff_min_area_ratio), 0.0)
    max_area_ratio = min(max(float(args.lseg_stuff_max_area_ratio), min_area_ratio), 1.0)
    kernel_size = max(int(args.lseg_stuff_morph_kernel), 0)
    morph_kernel = (
        np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if kernel_size > 1
        else None
    )
    batch_size = max(int(args.lseg_stuff_batch_size), 1)
    max_side = int(args.lseg_stuff_max_side)
    use_amp = bool(args.lseg_stuff_amp) and str(device).startswith("cuda")
    t0 = time.time()
    print(
        f"LSeg STUFF in-process running labels={labels} background={background_labels} "
        f"frames={len(image_paths)} stride={stride} batch={batch_size} "
        f"max_side={max_side} device={device}",
        flush=True,
    )

    frame_jobs = [
        (int(frame_idx), image_path)
        for frame_idx, image_path in enumerate(image_paths)
        if int(frame_idx) % stride == 0
    ]
    all_probs: List[Tuple[int, np.ndarray]] = []
    with torch.inference_mode():
        for batch_start in range(0, len(frame_jobs), batch_size):
            batch_jobs = frame_jobs[batch_start:batch_start + batch_size]
            tensors: List[torch.Tensor] = []
            for _frame_idx, image_path in batch_jobs:
                tensor, (frame_h, frame_w) = _read_lseg_rgb_tensor(image_path, max_side=max_side)
                if frame_h != H or frame_w != W:
                    raise ValueError(
                        f"LSeg STUFF frame size changed at {image_path}: "
                        f"{frame_h}x{frame_w} vs {H}x{W}"
                    )
                tensors.append(tensor)
            batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
            amp_context = (
                torch.autocast("cuda", dtype=torch.float16)
                if use_amp
                else contextlib.nullcontext()
            )
            with amp_context:
                logits = model(batch, labelset=all_prompts)
                if logits.shape[-2:] != (H, W):
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()
            for local_idx, (frame_idx, _image_path) in enumerate(batch_jobs):
                all_probs.append((int(frame_idx), probs[local_idx]))
            last_frame_idx = int(batch_jobs[-1][0])
            if (last_frame_idx + 1) % 100 == 0 or batch_start + batch_size >= len(frame_jobs):
                print(
                    f"  LSeg STUFF processed {last_frame_idx + 1}/{len(image_paths)} frames",
                    flush=True,
                )

    dense_probs = np.zeros((len(image_paths), len(all_prompts), H, W), dtype=np.float32)
    for frame_idx, frame_probs in all_probs:
        dense_probs[int(frame_idx)] = frame_probs
    tracks, counts = _lseg_payload_from_probs(
        dense_probs,
        label_prompts,
        image_paths,
        H=H,
        W=W,
        threshold=threshold,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        morph_kernel=morph_kernel,
    )
    return {
        "format": "lseg_stuff_tracks_v1",
        "num_frames": len(image_paths),
        "frame_height": H,
        "frame_width": W,
        "tracks": tracks,
        "debug": {
            "lseg_stuff_repo_root": repo_root,
            "lseg_stuff_checkpoint": os.path.abspath(os.path.expanduser(str(args.lseg_stuff_checkpoint))),
            "lseg_stuff_labels": labels,
            "lseg_stuff_background_labels": background_labels,
            "lseg_stuff_text_prompts": all_prompts,
            "lseg_stuff_prompt_template": prompt_template,
            "lseg_stuff_stride": int(stride),
            "lseg_stuff_batch_size": int(batch_size),
            "lseg_stuff_max_side": int(max_side),
            "lseg_stuff_threshold": float(threshold),
            "lseg_stuff_elapsed_seconds": float(time.time() - t0),
            "lseg_stuff_inprocess_seconds": float(time.time() - t0),
            **counts,
        },
    }


def _run_lseg_stuff_worker(args: argparse.Namespace) -> None:
    frame_list_path = args.lseg_stuff_frame_list
    output_pt = args.lseg_stuff_output_pt
    if not frame_list_path or not output_pt:
        raise ValueError("LSeg STUFF worker requires frame list and output path.")
    with open(frame_list_path, "r", encoding="utf-8") as handle:
        image_paths = [line.rstrip("\n") for line in handle if line.rstrip("\n")]
    payload = _run_lseg_stuff_inprocess_payload(image_paths, args)
    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(payload, output_pt)
    debug = payload.get("debug", {})
    print(
        f"LSeg STUFF worker saved {debug.get('lseg_stuff_tracks_added', 0)} tracks, "
        f"{debug.get('lseg_stuff_masks_added', 0)} frame masks -> {output_pt}",
        flush=True,
    )


def _run_lseg_stuff_subprocess_payload(
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
    *,
    prefix: str = "loger_lseg_stuff_",
) -> Dict:
    labels = _parse_comma_list(args.lseg_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return {"tracks": [], "debug": {"lseg_stuff_skipped": "disabled_or_no_labels"}}

    work_dir = tempfile.mkdtemp(prefix=prefix)
    temp_dirs.append(work_dir)
    frame_list_path = os.path.join(work_dir, "frames.txt")
    output_pt = os.path.join(work_dir, "stuff_tracks.pt")
    with open(frame_list_path, "w", encoding="utf-8") as handle:
        for path in image_paths:
            handle.write(f"{path}\n")

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--lseg_stuff_worker", "1",
        "--input", "__lseg_stuff_worker__",
        "--lseg_stuff_frame_list", frame_list_path,
        "--lseg_stuff_output_pt", output_pt,
        "--device", str(args.device),
        "--efficientsam3_stuff_enable", "1",
        "--efficientsam3_stuff_stride", str(args.efficientsam3_stuff_stride),
        "--lseg_stuff_repo_root", str(args.lseg_stuff_repo_root),
        "--lseg_stuff_checkpoint", str(args.lseg_stuff_checkpoint),
        "--lseg_stuff_prompts", str(args.lseg_stuff_prompts),
        "--lseg_stuff_prompt_template", str(args.lseg_stuff_prompt_template),
        "--lseg_stuff_background_prompts", str(args.lseg_stuff_background_prompts),
        "--lseg_stuff_confidence_threshold", str(args.lseg_stuff_confidence_threshold),
        "--lseg_stuff_min_area_ratio", str(args.lseg_stuff_min_area_ratio),
        "--lseg_stuff_max_area_ratio", str(args.lseg_stuff_max_area_ratio),
        "--lseg_stuff_morph_kernel", str(args.lseg_stuff_morph_kernel),
        "--lseg_stuff_batch_size", str(args.lseg_stuff_batch_size),
        "--lseg_stuff_max_side", str(args.lseg_stuff_max_side),
        "--lseg_stuff_half_res", str(args.lseg_stuff_half_res),
        "--lseg_stuff_amp", str(args.lseg_stuff_amp),
        "--lseg_stuff_device", str(args.lseg_stuff_device),
    ]

    print(
        "Running LSeg STUFF subprocess "
        f"(labels={labels}, stride={args.efficientsam3_stuff_stride}, "
        f"batch={args.lseg_stuff_batch_size}) ...",
        flush=True,
    )
    t0 = time.time()
    subprocess.run(cmd, check=True)
    payload = torch.load(output_pt, map_location="cpu", weights_only=False)
    payload.setdefault("debug", {})["lseg_stuff_subprocess_seconds"] = float(time.time() - t0)
    return payload


def _augment_with_lseg_stuff_subprocess(
    mo: SparseMaskletOutput,
    image_paths: List[str],
    args: argparse.Namespace,
    temp_dirs: List[str],
) -> SparseMaskletOutput:
    labels = _parse_comma_list(args.lseg_stuff_prompts)
    if not labels or not bool(args.efficientsam3_stuff_enable):
        return mo

    t0 = time.time()
    payload = _run_lseg_stuff_subprocess_payload(
        image_paths,
        args,
        temp_dirs,
    )
    augmented = _merge_efficientsam3_stuff_tracks(
        mo,
        payload,
        replace_existing=bool(args.efficientsam3_stuff_replace_existing),
        subtract_things=bool(args.efficientsam3_stuff_subtract_things),
        subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
    )
    debug = payload.get("debug", {})
    print(
        f"LSeg STUFF merge done in {time.time() - t0:.1f}s: "
        f"+{debug.get('lseg_stuff_tracks_added', 0)} tracks, "
        f"+{debug.get('lseg_stuff_masks_added', 0)} masks",
        flush=True,
    )
    return augmented


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


def _masklet_output_to_sparse_local(mo: MaskletOutput) -> SparseMaskletOutput:
    """Represent one chunk output as sparse local-frame tracks for STUFF merging."""
    tracks: List[Dict] = []
    for j in range(mo.num_masklets):
        birth_frame = int(mo.birth_frame[j]) if j < len(mo.birth_frame) else 0
        track = _make_empty_track(mo.num_frames, mo.frame_height, mo.frame_width, mo, j, birth_frame)
        _write_local_track_into_global(
            track,
            mo,
            j,
            global_start=0,
            total_frames=mo.num_frames,
            preserve_overlap_until=0,
        )
        if track["mask_by_frame"]:
            tracks.append(track)
    return SparseMaskletOutput(
        tracks=tracks,
        num_masklets=len(tracks),
        num_frames=mo.num_frames,
        frame_height=mo.frame_height,
        frame_width=mo.frame_width,
        debug=dict(getattr(mo, "debug", {}) or {}),
    )


def _merge_chunk_sparse_tracks_into_global(
    global_tracks: List[Dict],
    local_sparse: SparseMaskletOutput,
    *,
    chunk_idx: int,
    start: int,
    end: int,
    total_frames: int,
    chunk_overlap: int,
    source_types: Optional[set] = None,
) -> Dict[str, object]:
    """Merge local sparse tracks, mainly per-frame STUFF, into global tracks."""
    source_types = source_types or {"stuff_static"}
    H = int(local_sparse.frame_height)
    W = int(local_sparse.frame_width)
    overlap = 0 if int(chunk_idx) == 0 else min(int(chunk_overlap), int(local_sparse.num_frames))
    preserve_overlap_until = int(start) + int(overlap)
    matched = 0
    created = 0
    written_masks = 0

    for local_track in local_sparse.tracks:
        source_type = str(local_track.get("source_type", "?"))
        if source_type not in source_types:
            continue
        label = str(local_track.get("L_sem", "?"))
        group = int(local_track.get("G_sem", label_to_group(label)))
        global_idx: Optional[int] = None
        for idx, track in enumerate(global_tracks):
            if str(track.get("source_type", "?")) != source_type:
                continue
            if int(track.get("G_sem", -999)) != group:
                continue
            if not _labels_compatible(str(track.get("L_sem", "?")), label):
                continue
            global_idx = int(idx)
            matched += 1
            break

        if global_idx is None:
            new_track = {
                "mask_by_frame": {},
                "box_by_frame": {},
                "q_by_frame": {},
                "area_by_frame": {},
                "L_sem": label,
                "G_sem": group,
                "W_sem": float(local_track.get("W_sem", DEFAULT_SEMANTIC_WEIGHTS.get(group, 0.15))),
                "source_type": source_type,
                "birth_frame": int(start) + int(local_track.get("birth_frame", 0)),
                "frame_height": H,
                "frame_width": W,
            }
            global_tracks.append(new_track)
            global_idx = len(global_tracks) - 1
            created += 1
        else:
            global_tracks[global_idx]["birth_frame"] = min(
                int(global_tracks[global_idx].get("birth_frame", start)),
                int(start) + int(local_track.get("birth_frame", 0)),
            )

        global_track = global_tracks[global_idx]
        for local_frame, packed in local_track.get("mask_by_frame", {}).items():
            global_frame = int(start) + int(local_frame)
            if global_frame >= int(total_frames) or global_frame >= int(end):
                continue
            if (
                global_frame < preserve_overlap_until
                and global_frame in global_track["mask_by_frame"]
            ):
                continue
            packed_np = np.asarray(packed, dtype=np.uint8)
            global_track["mask_by_frame"][global_frame] = packed_np.copy()
            box = local_track.get("box_by_frame", {}).get(local_frame)
            if box is None:
                mask = _unpack_mask_np(packed_np, H, W)
                box = torch.from_numpy(_mask_to_box_np(mask))
                area = float(mask.sum()) / max(float(H * W), 1.0)
            else:
                box = box.clone() if torch.is_tensor(box) else torch.as_tensor(box)
                area = float(local_track.get("area_by_frame", {}).get(local_frame, 0.0))
            global_track["box_by_frame"][global_frame] = box
            global_track["q_by_frame"][global_frame] = float(
                local_track.get("q_by_frame", {}).get(local_frame, 0.0)
            )
            global_track["area_by_frame"][global_frame] = area
            written_masks += 1

    return {
        "chunk_index": int(chunk_idx),
        "frame_range": (int(start), int(end)),
        "num_local_sparse_tracks": int(local_sparse.num_masklets),
        "num_sparse_tracks_matched": int(matched),
        "num_sparse_tracks_created": int(created),
        "num_sparse_masks_written": int(written_masks),
        "source_types": sorted(source_types),
    }


def _build_chunk_seed_detections(
    global_tracks: List[Dict],
    curr_start: int,
    overlap: int,
    seed_carry_gap: int = 0,
) -> Dict[int, List[Dict]]:
    if overlap <= 0:
        return {}

    seed_dets: Dict[int, List[Dict]] = {}
    for global_idx, track in enumerate(global_tracks):
        source_type = track.get("source_type", "?")
        if source_type not in {"thing_tracked", "structure_tracked"}:
            continue

        produced_exact_overlap_seed = False
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
            sem_group = int(track.get("G_sem", -1))
            conf = float(track["q_by_frame"][global_t])
            area_ratio = float(track["area_by_frame"][global_t])

            seed_dets.setdefault(local_curr_t, []).append({
                "mask": mask.astype(np.uint8),
                "box": box,
                "confidence": conf,
                "label": label,
                "raw_label": label,
                "sem_group": sem_group,
                "area_ratio": area_ratio,
                "is_seed_track": True,
                "seed_global_track_idx": int(global_idx),
            })
            produced_exact_overlap_seed = True

        if (
            produced_exact_overlap_seed
            or int(seed_carry_gap) <= 0
            or source_type != "thing_tracked"
        ):
            continue

        # A short SAM2Long-style handoff memory: if the previous chunk loses
        # the object exactly in the overlap, seed local frame 0 with the most
        # recent trusted THING mask instead of immediately minting a new ID.
        recent_start = max(0, int(curr_start) - int(seed_carry_gap))
        recent_frames = [
            int(t)
            for t in track["mask_by_frame"].keys()
            if recent_start <= int(t) < int(curr_start)
        ]
        if not recent_frames:
            continue
        src_t = max(recent_frames)
        mask = _unpack_mask_np(
            track["mask_by_frame"][src_t],
            int(track["frame_height"]),
            int(track["frame_width"]),
        )
        if mask.sum() <= 0:
            continue
        box = track["box_by_frame"][src_t].numpy().astype(np.float32)
        label = str(track.get("L_sem", "unknown"))
        sem_group = int(track.get("G_sem", -1))
        conf = 0.80 * float(track["q_by_frame"].get(src_t, track.get("W_sem", 0.0)))
        area_ratio = float(track["area_by_frame"].get(src_t, 0.0))
        seed_dets.setdefault(0, []).append({
            "mask": mask.astype(np.uint8),
            "box": box,
            "confidence": conf,
            "label": label,
            "raw_label": label,
            "sem_group": sem_group,
            "area_ratio": area_ratio,
            "is_seed_track": True,
            "is_carry_seed_track": True,
            "seed_carry_source_frame": int(src_t),
            "seed_global_track_idx": int(global_idx),
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


def _cli_arg_provided(argv: List[str], name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def apply_sam31_route_preset(args: argparse.Namespace, argv: List[str]) -> None:
    """Apply named route defaults without blocking explicit CLI overrides."""
    route = str(getattr(args, "sam31_route", "custom")).strip().lower()
    if route not in {"b_quality"}:
        return

    def set_if_missing(flag: str, attr: str, value) -> None:
        if not _cli_arg_provided(argv, flag):
            setattr(args, attr, value)

    # Route B: SAM3.1 text prompts own THING discovery/tracking; YOLOE-11l only
    # supplies low-res support masks for prompt-frame selection and person repair.
    set_if_missing("--sam_backend", "sam_backend", "sam31_multiplex")
    set_if_missing("--detector", "detector", "yoloe")
    set_if_missing("--yoloe_model", "yoloe_model", "yoloe-11l-seg.pt")
    set_if_missing("--yoloe_imgsz", "yoloe_imgsz", 0)
    set_if_missing("--detector_max_side", "detector_max_side", 0)
    set_if_missing("--processing_max_side", "processing_max_side", 720)
    set_if_missing("--disable_stuff_prompts", "disable_stuff_prompts", 1)
    set_if_missing("--efficientsam3_stuff_enable", "efficientsam3_stuff_enable", 0)
    set_if_missing(
        "--sam31_text_track_labels",
        "sam31_text_track_labels",
        "person",
    )
    set_if_missing(
        "--sam31_direct_text_prompt_labels",
        "sam31_direct_text_prompt_labels",
        "person",
    )
    set_if_missing("--sam31_direct_text_prompt_frame_count", "sam31_direct_text_prompt_frame_count", 1)
    set_if_missing("--sam31_person_refresh_prompt_frames", "sam31_person_refresh_prompt_frames", 2)
    set_if_missing("--sam31_nontext_sparse_support", "sam31_nontext_sparse_support", 1)
    set_if_missing("--sam31_nontext_object_prompt_budget", "sam31_nontext_object_prompt_budget", 0)
    set_if_missing("--sam31_max_text_prompt_objects", "sam31_max_text_prompt_objects", 12)
    set_if_missing("--sam31_max_internal_objects", "sam31_max_internal_objects", 16)
    set_if_missing("--sam31_min_chunk_size", "sam31_min_chunk_size", 96)
    set_if_missing("--max_thing_objects", "max_thing_objects", 16)


def _compute_local_track_seed_priority(
    mo: MaskletOutput,
    local_idx: int,
    overlap: int,
) -> Tuple[int, int, float, float, int]:
    overlap_vis = 0
    overlap_mean_area = 0.0
    overlap_len = min(max(int(overlap), 0), mo.num_frames)
    if overlap_len > 0:
        overlap_visible = mo.V_mask[local_idx, :overlap_len]
        overlap_vis = int(overlap_visible.sum().item())
        if overlap_vis > 0:
            overlap_mean_area = float(
                mo.A_ratio[local_idx, :overlap_len][overlap_visible].mean().item()
            )

    total_vis = int(mo.V_mask[local_idx].sum().item())
    mean_area = (
        float(mo.A_ratio[local_idx, mo.V_mask[local_idx]].mean().item())
        if total_vis > 0 else 0.0
    )
    birth_frame = int(mo.birth_frame[local_idx]) if local_idx < len(mo.birth_frame) else 0
    return overlap_vis, total_vis, overlap_mean_area, mean_area, -birth_frame


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
    if global_track["source_type"] == "thing_tracked":
        return (
            0.40 * mean_iou
            + 0.25 * mean_cover
            + 0.15 * union_iou
            + 0.15 * mean_box_iou
            + 0.05 * mean_area_sim
        )
    if frame_ious:
        return 0.7 * mean_iou + 0.3 * union_iou
    return union_iou


def _compute_structure_track_similarity(track_a: Dict, track_b: Dict, total_frames: int) -> float:
    if track_a["source_type"] != "stuff_static" or track_b["source_type"] != "stuff_static":
        return -1.0
    if int(track_a["G_sem"]) != int(track_b["G_sem"]):
        return -1.0
    if not _labels_compatible(track_a["L_sem"], track_b["L_sem"]):
        return -1.0

    frame_box_ious: List[float] = []
    frame_area_sims: List[float] = []

    shared_frames = sorted(set(track_a["mask_by_frame"]).intersection(track_b["mask_by_frame"]))
    shared_frames = _sample_sorted_frames(shared_frames, 8)
    for t in shared_frames:
        frame_box_ious.append(_box_iou_xyxy_torch(track_a["box_by_frame"][t], track_b["box_by_frame"][t]))
        area_a = float(track_a["area_by_frame"][t])
        area_b = float(track_b["area_by_frame"][t])
        if area_a > 0.0 and area_b > 0.0:
            frame_area_sims.append(min(area_a, area_b) / max(area_a, area_b))

    if not shared_frames:
        # Stuff is semantic region support, not instance identity. If the same
        # stuff label appears in different chunks, merge it into one sparse
        # timeline instead of minting a new ID for every chunk.
        return 0.56

    mean_box_iou = sum(frame_box_ious) / len(frame_box_ious) if frame_box_ious else 0.0
    mean_area_sim = sum(frame_area_sims) / len(frame_area_sims) if frame_area_sims else 0.0
    if mean_box_iou < 0.18 and mean_area_sim < 0.35:
        return -1.0
    return 0.70 * mean_box_iou + 0.30 * mean_area_sim


def _track_visible_frames(track: Dict) -> List[int]:
    return sorted(int(t) for t in track["mask_by_frame"].keys())


def _canonical_track_label(label: str) -> str:
    label_l = str(label).strip().lower()
    if label_l in {"person", "people"}:
        return "person"
    return label_l


def _track_candidate_key(track: Dict) -> Tuple[str, int, str]:
    return (
        str(track.get("source_type", "?")),
        int(track.get("G_sem", -1)),
        _canonical_track_label(str(track.get("L_sem", ""))),
    )


def _track_frame_bounds(track: Dict) -> Optional[Tuple[int, int]]:
    frames = track.get("mask_by_frame", {})
    if not frames:
        return None
    keys = [int(t) for t in frames.keys()]
    return min(keys), max(keys)


def _track_temporal_match_possible(track_a: Dict, track_b: Dict) -> bool:
    bounds_a = _track_frame_bounds(track_a)
    bounds_b = _track_frame_bounds(track_b)
    if bounds_a is None or bounds_b is None:
        return False
    a0, a1 = bounds_a
    b0, b1 = bounds_b
    if max(a0, b0) <= min(a1, b1):
        return True

    if a1 < b0:
        gap = b0 - a1 - 1
    else:
        gap = a0 - b1 - 1
    label = _canonical_track_label(str(track_a.get("L_sem", "")))
    source_type = str(track_a.get("source_type", "?"))
    if source_type == "structure_tracked":
        max_gap = 14
    elif label == "person":
        max_gap = 36
    else:
        max_gap = 10
    return gap <= max_gap


def _group_track_indices_for_matching(global_tracks: List[Dict], removed: set) -> Dict[Tuple[str, int, str], List[int]]:
    groups: Dict[Tuple[str, int, str], List[int]] = {}
    for idx, track in enumerate(global_tracks):
        if idx in removed:
            continue
        source_type = str(track.get("source_type", "?"))
        if source_type not in {"thing_tracked", "structure_tracked", "stuff_static"}:
            continue
        groups.setdefault(_track_candidate_key(track), []).append(idx)
    return groups


def _sample_sorted_frames(frames: List[int], max_frames: int) -> List[int]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    positions = np.linspace(0, len(frames) - 1, int(max_frames))
    return [frames[int(round(float(pos)))] for pos in positions]


def _mean_box_and_area(track: Dict, frames: List[int]) -> Tuple[torch.Tensor, float]:
    boxes = torch.stack([track["box_by_frame"][int(t)].float() for t in frames], dim=0)
    mean_box = boxes.mean(dim=0)
    mean_area = float(
        sum(float(track["area_by_frame"][int(t)]) for t in frames) / max(len(frames), 1)
    )
    return mean_box, mean_area


def _box_center_size(box: torch.Tensor) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), w, h


def _compute_temporal_gap_track_similarity(track_a: Dict, track_b: Dict) -> float:
    if track_a["source_type"] != track_b["source_type"]:
        return -1.0
    if track_a["source_type"] not in {"thing_tracked", "structure_tracked"}:
        return -1.0
    if int(track_a["G_sem"]) != int(track_b["G_sem"]):
        return -1.0
    if not _labels_compatible(track_a["L_sem"], track_b["L_sem"]):
        return -1.0

    frames_a = _track_visible_frames(track_a)
    frames_b = _track_visible_frames(track_b)
    if not frames_a or not frames_b:
        return -1.0

    if frames_a[-1] < frames_b[0]:
        early, late = track_a, track_b
        early_frames, late_frames = frames_a, frames_b
    elif frames_b[-1] < frames_a[0]:
        early, late = track_b, track_a
        early_frames, late_frames = frames_b, frames_a
    else:
        return -1.0

    gap = int(late_frames[0] - early_frames[-1] - 1)
    label = str(track_a.get("L_sem", "")).strip().lower()
    is_person = _labels_compatible(label, "person") or _labels_compatible(label, "people")
    is_structure = track_a["source_type"] == "structure_tracked"
    max_gap = 36 if is_person else (14 if is_structure else 10)
    if gap < 0 or gap > max_gap:
        return -1.0
    if min(len(early_frames), len(late_frames)) < 2 and gap > (6 if is_structure else 3):
        return -1.0

    edge = 4 if is_structure else 3
    early_edge = early_frames[-edge:]
    late_edge = late_frames[:edge]
    early_box, early_area = _mean_box_and_area(early, early_edge)
    late_box, late_area = _mean_box_and_area(late, late_edge)
    if early_area <= 0.0 or late_area <= 0.0:
        return -1.0

    e_cx, e_cy, e_w, e_h = _box_center_size(early_box)
    l_cx, l_cy, l_w, l_h = _box_center_size(late_box)
    center_dist = float(np.hypot(l_cx - e_cx, l_cy - e_cy))
    avg_diag = 0.5 * (float(np.hypot(e_w, e_h)) + float(np.hypot(l_w, l_h)))
    center_norm = center_dist / max(avg_diag, 1.0)

    area_sim = min(early_area, late_area) / max(early_area, late_area)
    width_sim = min(e_w, l_w) / max(e_w, l_w)
    height_sim = min(e_h, l_h) / max(e_h, l_h)
    size_sim = 0.5 * (width_sim + height_sim)

    if is_structure:
        allowed_center = min(2.25 + 0.06 * float(gap), 3.00)
        min_area_sim = 0.16
        min_size_sim = 0.18
    else:
        allowed_center = 1.25 + 0.04 * float(gap) if is_person else 0.95 + 0.03 * float(gap)
        allowed_center = min(allowed_center, 2.35 if is_person else 1.40)
        min_area_sim = 0.25 if is_person else 0.40
        min_size_sim = 0.25 if is_person else 0.40
    if center_norm > allowed_center or area_sim < min_area_sim or size_sim < min_size_sim:
        return -1.0

    center_score = 1.0 - min(center_norm / max(allowed_center, 1e-6), 1.0)
    gap_score = 1.0 - min(float(gap) / max(float(max_gap), 1.0), 1.0)
    if is_structure:
        return (
            0.22 * center_score
            + 0.38 * area_sim
            + 0.28 * size_sim
            + 0.12 * gap_score
        )
    return (
        0.42 * center_score
        + 0.32 * area_sim
        + 0.18 * size_sim
        + 0.08 * gap_score
    )


def _compute_global_track_similarity(track_a: Dict, track_b: Dict) -> float:
    if track_a["source_type"] != track_b["source_type"]:
        return -1.0
    if int(track_a["G_sem"]) != int(track_b["G_sem"]):
        return -1.0
    if not _labels_compatible(track_a["L_sem"], track_b["L_sem"]):
        return -1.0

    source_type = track_a["source_type"]
    if source_type not in {"thing_tracked", "structure_tracked"}:
        return -1.0

    shared_frames = sorted(set(track_a["mask_by_frame"]).intersection(track_b["mask_by_frame"]))
    shared_frames = _sample_sorted_frames(shared_frames, 8)
    if len(shared_frames) < 1:
        return -1.0

    frame_box_ious: List[float] = []
    frame_area_sims: List[float] = []
    frame_center_scores: List[float] = []

    for t in shared_frames:
        box_a = track_a["box_by_frame"][t]
        box_b = track_b["box_by_frame"][t]
        frame_box_ious.append(_box_iou_xyxy_torch(box_a, box_b))
        center_norm = _boxes_center_norm(box_a, box_b)
        frame_center_scores.append(max(0.0, 1.0 - min(center_norm, 1.6) / 1.6))
        area_a = float(track_a["area_by_frame"][t])
        area_b = float(track_b["area_by_frame"][t])
        if area_a > 0.0 and area_b > 0.0:
            frame_area_sims.append(min(area_a, area_b) / max(area_a, area_b))

    mean_box_iou = sum(frame_box_ious) / len(frame_box_ious) if frame_box_ious else 0.0
    mean_area_sim = sum(frame_area_sims) / len(frame_area_sims) if frame_area_sims else 0.0
    mean_center_score = sum(frame_center_scores) / len(frame_center_scores) if frame_center_scores else 0.0
    label = _canonical_track_label(str(track_a.get("L_sem", "")))
    is_person = label == "person"

    if len(shared_frames) == 1:
        if source_type == "thing_tracked":
            min_box = 0.70 if is_person else 0.74
            min_area = 0.50 if is_person else 0.58
            if mean_box_iou < min_box or mean_area_sim < min_area:
                return -1.0
            return 0.60 * mean_box_iou + 0.25 * mean_area_sim + 0.15 * mean_center_score
        if mean_box_iou < 0.65 or mean_area_sim < 0.45:
            return -1.0
        return 0.65 * mean_box_iou + 0.25 * mean_area_sim + 0.10 * mean_center_score

    if source_type == "thing_tracked":
        if is_person:
            if mean_box_iou < 0.32 and mean_center_score < 0.58:
                return -1.0
            if mean_area_sim < 0.28:
                return -1.0
        else:
            if mean_box_iou < 0.45 or mean_area_sim < 0.40:
                return -1.0
        return (
            0.55 * mean_box_iou
            + 0.25 * mean_area_sim
            + 0.20 * mean_center_score
        )

    if mean_box_iou < 0.38 and mean_center_score < 0.55:
        return -1.0
    return (
        0.60 * mean_box_iou
        + 0.25 * mean_area_sim
        + 0.15 * mean_center_score
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
        secondary_q = float(secondary["q_by_frame"][t])
        primary_q = float(primary["q_by_frame"][t])
        secondary_area = float(secondary["area_by_frame"].get(t, 0.0))
        primary_area = float(primary["area_by_frame"].get(t, 0.0))
        if secondary_q > primary_q or (
            abs(secondary_q - primary_q) <= 1e-6
            and secondary_area > primary_area
        ):
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
        groups = _group_track_indices_for_matching(global_tracks, removed)
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue
            for pos, i in enumerate(group_indices):
                for j in group_indices[pos + 1:]:
                    if not _track_temporal_match_possible(global_tracks[i], global_tracks[j]):
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


def _deduplicate_thing_structure_tracks(
    global_tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    removed = set()
    merge_debug: List[Dict[str, float]] = []

    while True:
        candidates: List[Tuple[float, int, int, str]] = []
        groups = _group_track_indices_for_matching(global_tracks, removed)
        for group_indices in groups.values():
            if len(group_indices) <= 1:
                continue
            for pos, i in enumerate(group_indices):
                for j in group_indices[pos + 1:]:
                    if not _track_temporal_match_possible(global_tracks[i], global_tracks[j]):
                        continue
                    match_kind = "overlap"
                    score = _compute_global_track_similarity(global_tracks[i], global_tracks[j])
                    if score < 0.0:
                        score = _compute_temporal_gap_track_similarity(global_tracks[i], global_tracks[j])
                        match_kind = "gap"
                        if score < 0.0:
                            continue
                    source_type = global_tracks[i]["source_type"]
                    if match_kind == "gap":
                        threshold = 0.50 if source_type == "thing_tracked" else 0.42
                    else:
                        threshold = 0.64 if source_type == "thing_tracked" else 0.66
                    if score < threshold:
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
                    candidates.append((score, primary, secondary, match_kind))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        merged_this_round = False
        used_this_round = set()
        for score, primary, secondary, match_kind in candidates:
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
                "kind": match_kind,
            })
            merged_this_round = True

        if not merged_this_round:
            break

    deduped = [track for idx, track in enumerate(global_tracks) if idx not in removed]
    return deduped, merge_debug


def _prune_weak_global_tracks(
    global_tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    kept: List[Dict] = []
    prune_debug: List[Dict[str, float]] = []

    for idx, track in enumerate(global_tracks):
        visible_frames = len(track["mask_by_frame"])
        mean_area = (
            float(sum(track["area_by_frame"].values()) / max(visible_frames, 1))
            if visible_frames > 0 else 0.0
        )
        source_type = str(track.get("source_type", "?"))
        label = str(track.get("L_sem", "")).strip().lower()

        drop = False
        if visible_frames <= 0:
            drop = True
        elif source_type == "thing_tracked":
            drop = visible_frames <= 1 or (
                visible_frames <= 3 and mean_area < 0.0015
            )
            if (
                not drop
                and any(_labels_compatible(label, key) for key in {"door", "window"})
                and mean_area > 0.10
            ):
                drop = True
            if not drop and _labels_compatible(label, "person"):
                drop = (
                    (visible_frames <= 4 and mean_area < 0.0020)
                    or (visible_frames <= 10 and mean_area < 0.0012)
                    or (visible_frames <= 16 and mean_area < 0.0020)
                    or (visible_frames <= 32 and mean_area < 0.0010)
                    or (visible_frames <= 48 and mean_area < 0.0008)
                )
            if (
                not drop
                and any(
                    _labels_compatible(label, key)
                    for key in {
                        "chair",
                        "monitor",
                        "laptop",
                        "book",
                        "bottle",
                        "mouse",
                        "keyboard",
                        "cup",
                        "phone",
                        "bicycle",
                    }
                )
            ):
                drop = (
                    (visible_frames <= 15 and mean_area < 0.0045)
                    or (visible_frames <= 35 and mean_area < 0.0025)
                )
        elif source_type == "structure_tracked":
            drop = visible_frames <= 1 and mean_area < 0.01
        elif source_type == "stuff_static":
            drop = visible_frames <= 1 and mean_area < 0.01

        if drop:
            prune_debug.append(
                {
                    "track_index": int(idx),
                    "visible_frames": int(visible_frames),
                    "mean_area": float(mean_area),
                    "label": str(label),
                }
            )
            continue

        kept.append(track)

    return kept, prune_debug


def _clean_sparse_track_masks(
    global_tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    clean_debug: List[Dict[str, float]] = []

    for idx, track in enumerate(global_tracks):
        if track.get("source_type") not in {"thing_tracked", "structure_tracked"}:
            continue
        H = int(track["frame_height"])
        W = int(track["frame_width"])
        image_area = max(H * W, 1)
        sem_group = int(track.get("G_sem", -1))
        label = str(track.get("L_sem", ""))

        for frame_idx in list(track["mask_by_frame"].keys()):
            mask = _unpack_mask_np(track["mask_by_frame"][frame_idx], H, W)
            before_area = int(mask.sum())
            cleaned = _clean_instance_mask_components(
                mask,
                sem_group,
                label,
                image_area=image_area,
            )
            after_area = int(cleaned.sum())
            if after_area <= 0:
                track["mask_by_frame"].pop(frame_idx, None)
                track["box_by_frame"].pop(frame_idx, None)
                track["q_by_frame"].pop(frame_idx, None)
                track["area_by_frame"].pop(frame_idx, None)
                continue
            if after_area != before_area:
                ys, xs = np.where(cleaned)
                track["mask_by_frame"][frame_idx] = _pack_mask_np(cleaned)
                track["box_by_frame"][frame_idx] = torch.tensor(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=torch.float32,
                )
                track["area_by_frame"][frame_idx] = float(after_area / image_area)
                if len(clean_debug) < 200:
                    clean_debug.append(
                        {
                            "track_index": int(idx),
                            "frame_idx": int(frame_idx),
                            "before_area": float(before_area / image_area),
                            "after_area": float(after_area / image_area),
                        }
                    )

        if track["mask_by_frame"]:
            track["birth_frame"] = min(int(t) for t in track["mask_by_frame"].keys())

    return global_tracks, clean_debug


def _regularize_structure_track_masks(
    global_tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, object]]]:
    """Drop propagated structure masks that no longer match their label geometry."""
    kept_tracks: List[Dict] = []
    debug: List[Dict[str, object]] = []

    for idx, track in enumerate(global_tracks):
        if int(track.get("G_sem", -1)) != SEMANTIC_GROUP_STRUCTURE_ANCHOR:
            kept_tracks.append(track)
            continue
        if track.get("source_type") not in {"structure_tracked", "stuff_static"}:
            kept_tracks.append(track)
            continue

        H = int(track.get("frame_height", 0))
        W = int(track.get("frame_width", 0))
        image_area = max(H * W, 1)
        label = str(track.get("L_sem", "")).strip().lower()
        removed_frames = 0
        reason_counts: Dict[str, int] = {}

        for frame_idx in list(track["mask_by_frame"].keys()):
            packed = track["mask_by_frame"].get(frame_idx)
            if packed is None:
                continue
            mask = _unpack_mask_np(packed, H, W)
            box = track["box_by_frame"].get(frame_idx)
            if isinstance(box, torch.Tensor):
                box_np = box.detach().cpu().numpy()
            elif box is None:
                ys, xs = np.where(mask)
                if xs.size == 0 or ys.size == 0:
                    box_np = None
                else:
                    box_np = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            else:
                box_np = np.asarray(box, dtype=np.float32)
            confidence = float(track["q_by_frame"].get(frame_idx, track.get("W_sem", 0.0)))
            area_ratio = float(track["area_by_frame"].get(frame_idx, float(mask.sum()) / image_area))
            keep, reason = passes_structure_mask_quality(
                label,
                mask,
                box_np,
                confidence,
                area_ratio,
                H,
                W,
            )
            if keep:
                continue
            track["mask_by_frame"].pop(frame_idx, None)
            track["box_by_frame"].pop(frame_idx, None)
            track["q_by_frame"].pop(frame_idx, None)
            track["area_by_frame"].pop(frame_idx, None)
            removed_frames += 1
            reason_counts[str(reason)] = int(reason_counts.get(str(reason), 0)) + 1

        dropped_track = False
        if track["mask_by_frame"]:
            track["birth_frame"] = min(int(t) for t in track["mask_by_frame"].keys())
            kept_tracks.append(track)
        else:
            dropped_track = True
            reason_counts["drop_empty_track"] = int(reason_counts.get("drop_empty_track", 0)) + 1

        if removed_frames > 0 or dropped_track:
            debug.append(
                {
                    "track_index": int(idx),
                    "label": label,
                    "removed_frames": int(removed_frames),
                    "remaining_frames": int(len(track["mask_by_frame"])),
                    "reasons": reason_counts,
                }
            )

    return kept_tracks, debug


def _bridge_short_track_gaps(
    global_tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    """Fill very short thing-track flicker gaps using endpoint interpolation.

    This is a lightweight analogue of SAM2Long's confidence-gated memory reuse:
    we only reuse an existing instance memory when the gap is tiny and the two
    endpoint boxes still look compatible. Stuff masks are never bridged, and
    person masks are warped between endpoint boxes instead of frozen in place.
    """
    bridge_debug: List[Dict[str, float]] = []

    for idx, track in enumerate(global_tracks):
        if track.get("source_type") != "thing_tracked":
            continue
        frames = sorted(int(t) for t in track["mask_by_frame"].keys())
        if len(frames) < 2:
            continue

        label = str(track.get("L_sem", "")).strip().lower()
        is_person = _labels_compatible(label, "person") or _labels_compatible(label, "people")
        # At the current Taylor verification FPS, 12 frames is about 1.2 seconds:
        # enough to cover short SAM3.1 flicker / occlusion drops without turning
        # this into long-horizon mask hallucination.
        max_gap = 12 if is_person else 3
        H = int(track.get("frame_height", 0))
        W = int(track.get("frame_width", 0))
        if H <= 0 or W <= 0:
            continue

        for left, right in zip(frames[:-1], frames[1:]):
            gap = int(right - left - 1)
            if gap <= 0 or gap > max_gap:
                continue
            if left < 0 or right >= total_frames:
                continue

            left_area = float(track["area_by_frame"].get(left, 0.0))
            right_area = float(track["area_by_frame"].get(right, 0.0))
            if left_area <= 0.0 or right_area <= 0.0:
                continue
            area_sim = min(left_area, right_area) / max(left_area, right_area)
            center_norm = _boxes_center_norm(
                track["box_by_frame"][left],
                track["box_by_frame"][right],
            )
            if is_person:
                if gap > 4:
                    min_area_sim = 0.22
                    max_center_norm = 1.15
                else:
                    min_area_sim = 0.30
                    max_center_norm = 1.35
                larger_area = max(left_area, right_area)
                asymmetric_person_gap = (
                    gap <= 8
                    and area_sim >= 0.035
                    and area_sim < min_area_sim
                    and center_norm <= min(max_center_norm, 0.85)
                    and larger_area >= 0.0035
                )
                if (
                    (area_sim < min_area_sim or center_norm > max_center_norm)
                    and not asymmetric_person_gap
                ):
                    continue
            else:
                asymmetric_person_gap = False
                if area_sim < 0.45 or center_norm > 0.95:
                    continue

            left_box = track["box_by_frame"][left].detach().cpu().numpy().astype(np.float32)
            right_box = track["box_by_frame"][right].detach().cpu().numpy().astype(np.float32)
            left_mask = _unpack_mask_np(track["mask_by_frame"][left], H, W)
            right_mask = _unpack_mask_np(track["mask_by_frame"][right], H, W)
            if left_mask.sum() <= 0 or right_mask.sum() <= 0:
                continue

            for frame_idx in range(left + 1, right):
                if frame_idx in track["mask_by_frame"]:
                    continue
                alpha = float(frame_idx - left) / float(gap + 1)
                target_box = (1.0 - alpha) * left_box + alpha * right_box
                if asymmetric_person_gap:
                    if left_area >= right_area:
                        src_mask, src_box = left_mask, left_box
                    else:
                        src_mask, src_box = right_mask, right_box
                    target_box = _enforce_min_box_scale(
                        target_box,
                        src_box,
                        min_scale=0.55,
                        H=H,
                        W=W,
                    )
                    interp_mask = _warp_mask_between_boxes(src_mask, src_box, target_box, H, W)
                elif alpha <= 0.5:
                    interp_mask = _warp_mask_between_boxes(left_mask, left_box, target_box, H, W)
                else:
                    interp_mask = _warp_mask_between_boxes(right_mask, right_box, target_box, H, W)
                interp_area = int(interp_mask.sum())
                if interp_area <= 0:
                    continue
                ys, xs = np.where(interp_mask)
                track["mask_by_frame"][frame_idx] = _pack_mask_np(interp_mask)
                track["box_by_frame"][frame_idx] = torch.tensor(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=torch.float32,
                )
                track["area_by_frame"][frame_idx] = float(interp_area / max(H * W, 1))
                track["q_by_frame"][frame_idx] = 0.75 * min(
                    float(track["q_by_frame"].get(left, 1.0)),
                    float(track["q_by_frame"].get(right, 1.0)),
                )
                if len(bridge_debug) < 200:
                    bridge_debug.append(
                        {
                            "track_index": int(idx),
                            "frame_idx": int(frame_idx),
                            "left": int(left),
                            "right": int(right),
                            "gap": int(gap),
                            "area_sim": float(area_sim),
                            "center_norm": float(center_norm),
                            "asymmetric": bool(asymmetric_person_gap),
                        }
                    )

        if track["mask_by_frame"]:
            track["birth_frame"] = min(int(t) for t in track["mask_by_frame"].keys())

    return global_tracks, bridge_debug


def _track_frame_priority(track: Dict, track_index: int, frame_idx: int) -> Tuple[int, float, float, int, int]:
    visible_frames = len(track["mask_by_frame"])
    mean_q = (
        float(sum(track["q_by_frame"].values()) / max(visible_frames, 1))
        if visible_frames > 0 else 0.0
    )
    mean_area = (
        float(sum(track["area_by_frame"].values()) / max(visible_frames, 1))
        if visible_frames > 0 else 0.0
    )
    area_at_frame = float(track["area_by_frame"].get(frame_idx, mean_area))
    # Prefer longer, higher-confidence tracks. Earlier births win ties so
    # chunk handoffs do not get displaced by late duplicate fragments.
    return (
        int(visible_frames),
        float(mean_q),
        float(area_at_frame),
        -int(track.get("birth_frame", 0)),
        -int(track_index),
    )


def _boxes_center_norm(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax, ay, aw, ah = _box_center_size(box_a)
    bx, by, bw, bh = _box_center_size(box_b)
    dist = float(np.hypot(ax - bx, ay - by))
    avg_diag = 0.5 * (float(np.hypot(aw, ah)) + float(np.hypot(bw, bh)))
    return dist / max(avg_diag, 1.0)


def _box_candidate_cover_xyxy_torch(box_candidate: torch.Tensor, box_reference: torch.Tensor) -> float:
    cx1, cy1, cx2, cy2 = [float(v) for v in box_candidate.tolist()]
    rx1, ry1, rx2, ry2 = [float(v) for v in box_reference.tolist()]
    inter_x1 = max(cx1, rx1)
    inter_y1 = max(cy1, ry1)
    inter_x2 = min(cx2, rx2)
    inter_y2 = min(cy2, ry2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    cand_area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return inter / cand_area if cand_area > 0.0 else 0.0


def _box_center_inside_expanded_reference(
    box_candidate: torch.Tensor,
    box_reference: torch.Tensor,
    *,
    expand: float = 0.08,
) -> bool:
    cx1, cy1, cx2, cy2 = [float(v) for v in box_candidate.tolist()]
    rx1, ry1, rx2, ry2 = [float(v) for v in box_reference.tolist()]
    rcx, rcy, rw, rh = _box_center_size(box_reference)
    ccx = 0.5 * (cx1 + cx2)
    ccy = 0.5 * (cy1 + cy2)
    return (
        rx1 - expand * rw <= ccx <= rx2 + expand * rw
        and ry1 - expand * rh <= ccy <= ry2 + expand * rh
    )


def _person_spatial_fragment_duplicate(
    cand_box: torch.Tensor,
    ref_box: torch.Tensor,
    *,
    cand_area_ratio: float,
    ref_area_ratio: float,
) -> bool:
    """Detect small person fragments nested inside a stronger person track.

    SAM2Long-style multi-branch recovery improves recall, but it can also leave
    small partial-person branches around a stable full-body branch. This helper
    only marks very small candidates as duplicates when box geometry agrees
    that the candidate is spatial support of the stronger track.
    """
    if cand_area_ratio <= 0.0 or ref_area_ratio <= 0.0:
        return False
    # Person prompt propagation often leaves torso/leg/arm fragments around a
    # stronger full-body track. Keep this bounded so nearby small people are not
    # erased just because they overlap in a crowded frame.
    max_fragment_area = 0.026 if ref_area_ratio >= 0.045 else 0.018
    if cand_area_ratio > max_fragment_area:
        return False
    if cand_area_ratio > 0.55 * ref_area_ratio:
        return False

    box_cover = _box_candidate_cover_xyxy_torch(cand_box, ref_box)
    box_iou = _box_iou_xyxy_torch(cand_box, ref_box)
    center_norm = _boxes_center_norm(cand_box, ref_box)
    center_inside = _box_center_inside_expanded_reference(cand_box, ref_box)

    return (
        (box_cover >= 0.42 and center_norm <= 1.25)
        or (box_iou >= 0.06 and center_norm <= 1.30)
        or (center_inside and box_cover >= 0.10 and center_norm <= 1.00)
        or (box_cover >= 0.18 and center_norm <= 0.78)
    )


def _suppress_duplicate_track_frames(
    global_tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    """Remove frame-level duplicate masks while keeping complementary spans.

    Global merging is intentionally conservative to avoid collapsing two nearby
    people. This pass is a safer cleanup: if two tracks of the same semantic
    label are visible on the same frame and one mask is mostly covered by a
    stronger/stabler track, only that duplicated frame is suppressed.
    """
    suppress_debug: List[Dict[str, float]] = []
    suppressed_by_track: Dict[int, int] = {}

    track_priorities = {
        idx: _track_frame_priority(track, idx, -1)
        for idx, track in enumerate(global_tracks)
    }

    for frame_idx in range(total_frames):
        label_to_indices: Dict[Tuple[str, int, str], List[int]] = {}
        for idx, track in enumerate(global_tracks):
            if track.get("source_type") != "thing_tracked":
                continue
            if frame_idx not in track["mask_by_frame"]:
                continue
            label = str(track.get("L_sem", "")).strip().lower()
            key = (label, int(track.get("G_sem", -1)), str(track.get("source_type", "?")))
            label_to_indices.setdefault(key, []).append(idx)

        for (group_label, sem_group, _), indices in label_to_indices.items():
            if len(indices) <= 1:
                continue

            is_person_group = (
                _labels_compatible(group_label, "person")
                or _labels_compatible(group_label, "people")
            )
            if is_person_group:
                # In crowded performer shots, long-lived partial-person tracks
                # can otherwise outrank a fuller same-frame person mask. For
                # frame-level suppression, current-frame completeness should be
                # the first signal; track length is only a stability tie-breaker.
                ordered = sorted(
                    indices,
                    key=lambda idx: (
                        float(global_tracks[idx]["area_by_frame"].get(frame_idx, 0.0)),
                        track_priorities[idx][0],
                        track_priorities[idx][1],
                        track_priorities[idx][3],
                        track_priorities[idx][4],
                    ),
                    reverse=True,
                )
            else:
                ordered = sorted(
                    indices,
                    key=lambda idx: (
                        track_priorities[idx][0],
                        track_priorities[idx][1],
                        float(global_tracks[idx]["area_by_frame"].get(frame_idx, 0.0)),
                        track_priorities[idx][3],
                        track_priorities[idx][4],
                    ),
                    reverse=True,
                )
            kept: List[Tuple[int, np.ndarray, torch.Tensor]] = []
            for idx in ordered:
                track = global_tracks[idx]
                cand_mask = _unpack_mask_np(
                    track["mask_by_frame"][frame_idx],
                    int(track["frame_height"]),
                    int(track["frame_width"]),
                )
                cand_area = float(cand_mask.sum())
                if cand_area <= 0.0:
                    continue
                cand_box = track["box_by_frame"][frame_idx]
                cand_label = str(track.get("L_sem", "")).strip().lower()
                is_person = _labels_compatible(cand_label, "person") or _labels_compatible(cand_label, "people")
                image_area = max(int(track["frame_height"]) * int(track["frame_width"]), 1)
                cand_area_ratio = cand_area / float(image_area)

                suppress = False
                suppress_against = None
                suppress_stats = (0.0, 0.0, 0.0, 0.0)
                for kept_idx, kept_mask, kept_box in kept:
                    kept_area = float(kept_mask.sum())
                    if kept_area <= 0.0:
                        continue
                    kept_area_ratio = kept_area / float(image_area)
                    inter = float((cand_mask & kept_mask).sum())
                    union = cand_area + kept_area - inter
                    iou = inter / union if union > 0.0 else 0.0
                    cand_cover = inter / max(cand_area, 1.0)
                    kept_cover = inter / max(kept_area, 1.0)
                    box_iou = _box_iou_xyxy_torch(cand_box, kept_box)
                    center_norm = _boxes_center_norm(cand_box, kept_box)
                    area_sim = min(cand_area_ratio, kept_area_ratio) / max(
                        cand_area_ratio, kept_area_ratio, 1e-6
                    )

                    if is_person:
                        duplicate = (
                            iou >= 0.45
                            or box_iou >= 0.38
                            or (
                                cand_cover >= 0.62
                                and (box_iou >= 0.10 or center_norm <= 1.25)
                            )
                            or (
                                cand_cover >= 0.50
                                and kept_cover >= 0.50
                                and box_iou >= 0.18
                                and area_sim >= 0.35
                            )
                            or (
                                center_norm <= 0.35
                                and area_sim >= 0.42
                                and (cand_cover >= 0.25 or kept_cover >= 0.25)
                            )
                        )
                        duplicate = duplicate or (
                            cand_cover >= 0.05
                            and _person_spatial_fragment_duplicate(
                                cand_box,
                                kept_box,
                                cand_area_ratio=float(cand_area_ratio),
                                ref_area_ratio=float(kept_area_ratio),
                            )
                        )
                    elif sem_group == 1:
                        duplicate = (
                            iou >= 0.48
                            or cand_cover >= 0.78
                            or (cand_cover >= 0.62 and box_iou >= 0.36)
                        )
                    else:
                        duplicate = (
                            iou >= 0.52
                            or (
                                cand_cover >= 0.80
                                and (box_iou >= 0.24 or center_norm <= 0.68)
                            )
                        )

                    if duplicate:
                        suppress = True
                        suppress_against = int(kept_idx)
                        suppress_stats = (iou, cand_cover, kept_cover, box_iou)
                        break

                if suppress:
                    track["mask_by_frame"].pop(frame_idx, None)
                    track["box_by_frame"].pop(frame_idx, None)
                    track["q_by_frame"].pop(frame_idx, None)
                    track["area_by_frame"].pop(frame_idx, None)
                    suppressed_by_track[idx] = suppressed_by_track.get(idx, 0) + 1
                    if len(suppress_debug) < 200:
                        iou, cand_cover, kept_cover, box_iou = suppress_stats
                        suppress_debug.append(
                            {
                                "track_index": int(idx),
                                "frame_idx": int(frame_idx),
                                "against": int(suppress_against) if suppress_against is not None else -1,
                                "iou": float(iou),
                                "candidate_cover": float(cand_cover),
                                "kept_cover": float(kept_cover),
                                "box_iou": float(box_iou),
                            }
                        )
                    continue

                kept.append((idx, cand_mask, cand_box))

    for idx, track in enumerate(global_tracks):
        if not track["mask_by_frame"]:
            continue
        track["birth_frame"] = min(int(t) for t in track["mask_by_frame"].keys())

    if suppressed_by_track:
        suppress_debug.append(
            {
                "num_tracks_suppressed": int(len(suppressed_by_track)),
                "num_frames_suppressed": int(sum(suppressed_by_track.values())),
            }
        )

    return global_tracks, suppress_debug


def _drop_redundant_global_tracks(
    global_tracks: List[Dict],
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    """Drop tracks whose support is mostly explained by stronger same-label tracks."""
    drop_debug: List[Dict[str, float]] = []
    removed = set()
    priorities = {
        idx: _track_frame_priority(track, idx, -1)
        for idx, track in enumerate(global_tracks)
    }
    mean_areas = {
        idx: (
            float(sum(track.get("area_by_frame", {}).values()) / max(len(track.get("area_by_frame", {})), 1))
            if track.get("area_by_frame") else 0.0
        )
        for idx, track in enumerate(global_tracks)
    }
    frame_sets = {
        idx: set(int(t) for t in track.get("mask_by_frame", {}).keys())
        for idx, track in enumerate(global_tracks)
    }
    groups = _group_track_indices_for_matching(global_tracks, removed=set())

    def _is_stronger(a: int, b: int) -> bool:
        label_a = str(global_tracks[a].get("L_sem", "")).strip().lower()
        label_b = str(global_tracks[b].get("L_sem", "")).strip().lower()
        is_person_pair = (
            (_labels_compatible(label_a, "person") or _labels_compatible(label_a, "people"))
            and (_labels_compatible(label_b, "person") or _labels_compatible(label_b, "people"))
        )
        if is_person_pair:
            return (
                mean_areas[a],
                priorities[a][0],
                priorities[a][1],
                priorities[a][3],
                priorities[a][4],
            ) > (
                mean_areas[b],
                priorities[b][0],
                priorities[b][1],
                priorities[b][3],
                priorities[b][4],
            )
        return priorities[a] > priorities[b]

    for idx, track in enumerate(global_tracks):
        if track.get("source_type") != "thing_tracked":
            continue
        frames = sorted(int(t) for t in track["mask_by_frame"].keys())
        if not frames:
            removed.add(idx)
            continue
        label = str(track.get("L_sem", "")).strip().lower()
        sem_group = int(track.get("G_sem", -1))
        is_person = _labels_compatible(label, "person") or _labels_compatible(label, "people")
        same_key_indices = groups.get(_track_candidate_key(track), [])
        same_label_stronger = [
            other_idx
            for other_idx in same_key_indices
            for other in [global_tracks[other_idx]]
            if other_idx != idx
            and other_idx not in removed
            and _is_stronger(other_idx, idx)
            and frame_sets[idx].intersection(frame_sets.get(other_idx, set()))
        ]
        if not same_label_stronger:
            continue
        same_label_stronger.sort(
            key=lambda other_idx: (
                len(frame_sets[idx].intersection(frame_sets.get(other_idx, set()))),
                mean_areas[other_idx] if is_person else priorities[other_idx][2],
                priorities[other_idx],
            ),
            reverse=True,
        )
        same_label_stronger = same_label_stronger[: (16 if is_person else 8)]

        duplicate_frames = 0
        checked_frames = 0
        sampled_frames = _sample_sorted_frames(frames, 48 if is_person else 32)
        for frame_idx in sampled_frames:
            cand_box = track["box_by_frame"][frame_idx]
            cand_area_ratio = float(track["area_by_frame"].get(frame_idx, 0.0))
            if cand_area_ratio <= 0.0:
                continue
            checked_frames += 1
            frame_duplicate = False

            for other_idx in same_label_stronger:
                other = global_tracks[other_idx]
                if frame_idx not in other["mask_by_frame"]:
                    continue
                other_box = other["box_by_frame"][frame_idx]
                box_iou = _box_iou_xyxy_torch(cand_box, other_box)
                center_norm = _boxes_center_norm(cand_box, other_box)
                if is_person:
                    if box_iou < 0.015 and center_norm > 1.70:
                        continue
                elif box_iou < 0.035 and center_norm > 1.15:
                    continue
                other_area_ratio = float(other["area_by_frame"].get(frame_idx, 0.0))
                if other_area_ratio <= 0.0:
                    continue
                area_sim = min(cand_area_ratio, other_area_ratio) / max(cand_area_ratio, other_area_ratio)
                cand_box_cover = _box_candidate_cover_xyxy_torch(cand_box, other_box)
                other_box_cover = _box_candidate_cover_xyxy_torch(other_box, cand_box)

                if is_person:
                    frame_duplicate = (
                        box_iou >= 0.34
                        or (cand_box_cover >= 0.58 and center_norm <= 1.25)
                        or (cand_box_cover >= 0.46 and other_box_cover >= 0.46 and area_sim >= 0.35)
                        or (center_norm <= 0.50 and area_sim >= 0.38)
                    )
                    frame_duplicate = frame_duplicate or (
                        _person_spatial_fragment_duplicate(
                            cand_box,
                            other_box,
                            cand_area_ratio=float(cand_area_ratio),
                            ref_area_ratio=float(other_area_ratio),
                        )
                    )
                else:
                    frame_duplicate = (
                        box_iou >= 0.50
                        or cand_box_cover >= 0.78
                        or (cand_box_cover >= 0.62 and area_sim >= 0.55)
                    )
                if frame_duplicate:
                    break

            if frame_duplicate:
                duplicate_frames += 1

        if checked_frames <= 0:
            removed.add(idx)
            continue

        duplicate_ratio = duplicate_frames / max(checked_frames, 1)
        visible_frames = len(frames)
        mean_area = (
            float(sum(track["area_by_frame"].values()) / max(visible_frames, 1))
            if visible_frames > 0 else 0.0
        )
        drop = duplicate_frames >= max(3, int(np.ceil(0.60 * checked_frames)))
        if is_person and visible_frames <= 120:
            drop = drop or duplicate_ratio >= 0.40
        if is_person and visible_frames <= 48:
            drop = drop or duplicate_ratio >= 0.32
        if is_person and visible_frames <= 96 and mean_area < 0.010:
            drop = drop or duplicate_ratio >= 0.30

        if drop:
            removed.add(idx)
            if len(drop_debug) < 200:
                drop_debug.append(
                    {
                        "track_index": int(idx),
                        "visible_frames": int(visible_frames),
                        "duplicate_frames": int(duplicate_frames),
                        "checked_frames": int(checked_frames),
                        "duplicate_ratio": float(duplicate_ratio),
                        "mean_area": float(mean_area),
                    }
                )

    return [track for idx, track in enumerate(global_tracks) if idx not in removed], drop_debug


def _drop_small_person_false_tracks(
    global_tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    """Remove short-lived tiny person false positives when a dominant person exists."""
    person_indices = [
        idx
        for idx, track in enumerate(global_tracks)
        if track.get("source_type") == "thing_tracked"
        and _labels_compatible(str(track.get("L_sem", "")), "person")
        and int(track.get("G_sem", -1)) == SEMANTIC_GROUP_MOVABLE_THING
        and track.get("mask_by_frame")
    ]
    if len(person_indices) <= 1:
        return global_tracks, []

    stats: Dict[int, Tuple[int, float, float, set]] = {}
    for idx in person_indices:
        track = global_tracks[idx]
        frames = {int(t) for t in track["mask_by_frame"].keys()}
        areas = [float(v) for v in track.get("area_by_frame", {}).values()]
        mean_area = float(sum(areas) / max(len(areas), 1)) if areas else 0.0
        max_area = float(max(areas)) if areas else 0.0
        stats[idx] = (len(frames), mean_area, max_area, frames)

    dominant_indices = [
        idx
        for idx, (vis, mean_area, _max_area, _frames) in stats.items()
        if vis >= max(12, int(round(0.45 * float(total_frames))))
        and mean_area >= 0.035
    ]
    if not dominant_indices:
        return global_tracks, []

    dominant_frames = set()
    dominant_mean_area = 0.0
    for idx in dominant_indices:
        _vis, mean_area, _max_area, frames = stats[idx]
        dominant_frames.update(frames)
        dominant_mean_area = max(dominant_mean_area, mean_area)

    removed = set()
    debug: List[Dict[str, float]] = []
    for idx in person_indices:
        if idx in dominant_indices:
            continue
        vis, mean_area, max_area, frames = stats[idx]
        if vis <= 0:
            continue
        overlap_ratio = len(frames.intersection(dominant_frames)) / max(float(vis), 1.0)
        tiny_relative_to_main = mean_area <= min(0.018, 0.25 * dominant_mean_area)
        short_lived = vis <= max(24, int(round(0.55 * float(total_frames))))
        brief_fragment = (
            vis <= 8
            and overlap_ratio >= 0.30
            and max_area <= 0.14
        )
        short_small_fragment = (
            vis <= 24
            and overlap_ratio >= 0.45
            and mean_area <= min(0.030, 0.65 * dominant_mean_area)
            and max_area <= 0.12
        )
        if (
            brief_fragment
            or short_small_fragment
            or (
                overlap_ratio >= 0.60
                and tiny_relative_to_main
                and short_lived
                and max_area <= 0.040
            )
        ):
            removed.add(idx)
            debug.append(
                {
                    "track_index": int(idx),
                    "visible_frames": int(vis),
                    "mean_area": float(mean_area),
                    "max_area": float(max_area),
                    "dominant_mean_area": float(dominant_mean_area),
                    "overlap_ratio": float(overlap_ratio),
                }
            )

    if not removed:
        return global_tracks, []
    return [track for idx, track in enumerate(global_tracks) if idx not in removed], debug


def _consolidate_primary_person_tracks(
    global_tracks: List[Dict],
    total_frames: int,
) -> Tuple[List[Dict], List[Dict[str, float]]]:
    """Merge the dominant single-person timeline across chunk breaks.

    Taylor-like egocentric/single-subject videos often have one salient person,
    but forward-only chunking can split that person into many local IDs. This
    pass merges tracks that are the largest person hypothesis on their frames,
    then removes tiny overlapping person hypotheses as likely false positives.
    """
    person_indices = [
        idx
        for idx, track in enumerate(global_tracks)
        if track.get("source_type") == "thing_tracked"
        and _labels_compatible(str(track.get("L_sem", "")), "person")
        and int(track.get("G_sem", -1)) == SEMANTIC_GROUP_MOVABLE_THING
        and track.get("mask_by_frame")
    ]
    if len(person_indices) <= 1:
        return global_tracks, []

    owner_counts: Dict[int, int] = {idx: 0 for idx in person_indices}
    for frame_idx in range(int(total_frames)):
        candidates: List[Tuple[float, float, int, int]] = []
        for idx in person_indices:
            track = global_tracks[idx]
            if frame_idx not in track["mask_by_frame"]:
                continue
            area_ratio = float(track["area_by_frame"].get(frame_idx, 0.0))
            if area_ratio < 0.010:
                continue
            q = float(track["q_by_frame"].get(frame_idx, 0.0))
            candidates.append((area_ratio, q, -int(track.get("birth_frame", 0)), idx))
        if not candidates:
            continue
        candidates.sort(reverse=True)
        owner_counts[int(candidates[0][-1])] += 1

    selected_indices = {
        idx
        for idx, count in owner_counts.items()
        if count >= max(6, int(round(0.12 * len(global_tracks[idx]["mask_by_frame"]))))
    }
    if not selected_indices:
        return global_tracks, []

    selected_owner_frames = sum(owner_counts[idx] for idx in selected_indices)
    if selected_owner_frames < max(24, int(round(0.08 * float(total_frames)))):
        return global_tracks, []

    primary_idx = max(
        selected_indices,
        key=lambda idx: (
            owner_counts[idx],
            len(global_tracks[idx]["mask_by_frame"]),
            sum(float(v) for v in global_tracks[idx]["area_by_frame"].values())
            / max(len(global_tracks[idx]["area_by_frame"]), 1),
            -int(global_tracks[idx].get("birth_frame", 0)),
        ),
    )

    removed = set()
    merge_debug: List[Dict[str, float]] = []
    for idx in sorted(selected_indices):
        if idx == primary_idx:
            continue
        _merge_global_track_payload(global_tracks[primary_idx], global_tracks[idx], total_frames)
        removed.add(idx)
        merge_debug.append(
            {
                "primary": int(primary_idx),
                "secondary": int(idx),
                "owner_frames": int(owner_counts[idx]),
                "visible_frames": int(len(global_tracks[idx]["mask_by_frame"])),
            }
        )

    primary_frames = {int(t) for t in global_tracks[primary_idx]["mask_by_frame"].keys()}
    primary_mean_area = (
        sum(float(v) for v in global_tracks[primary_idx]["area_by_frame"].values())
        / max(len(global_tracks[primary_idx]["area_by_frame"]), 1)
    )

    for idx in person_indices:
        if idx == primary_idx or idx in removed:
            continue
        track = global_tracks[idx]
        frames = {int(t) for t in track["mask_by_frame"].keys()}
        if not frames:
            removed.add(idx)
            continue
        areas = [float(v) for v in track["area_by_frame"].values()]
        mean_area = sum(areas) / max(len(areas), 1)
        max_area = max(areas) if areas else 0.0
        overlap_ratio = len(frames.intersection(primary_frames)) / max(float(len(frames)), 1.0)
        short_or_tiny = (
            len(frames) <= max(32, int(round(0.08 * float(total_frames))))
            or mean_area <= min(0.028, 0.45 * primary_mean_area)
            or max_area <= 0.040
        )
        if overlap_ratio >= 0.80 or (overlap_ratio >= 0.35 and short_or_tiny):
            removed.add(idx)
            merge_debug.append(
                {
                    "primary": int(primary_idx),
                    "secondary": int(idx),
                    "dropped_overlap_ratio": float(overlap_ratio),
                    "visible_frames": int(len(frames)),
                    "mean_area": float(mean_area),
                    "max_area": float(max_area),
                }
            )

    if not removed:
        return global_tracks, []

    kept = [track for idx, track in enumerate(global_tracks) if idx not in removed]
    return kept, merge_debug


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

    seeded_locals = set()
    rejected_seeded_locals = set()
    if getattr(mo, "seed_global_track_idx", None):
        best_seeded_local: Dict[int, int] = {}
        best_seeded_priority: Dict[int, Tuple[int, int, float, float, int]] = {}
        for j, seeded_global_idx in enumerate(mo.seed_global_track_idx):
            if seeded_global_idx is None:
                continue
            seeded_global_idx = int(seeded_global_idx)
            if not (0 <= seeded_global_idx < len(global_tracks)):
                continue
            local_type = mo.source_type[j] if j < len(mo.source_type) else "?"
            local_group = int(mo.G_sem[j].item())
            local_label = mo.L_sem[j] if j < len(mo.L_sem) else "?"
            track = global_tracks[seeded_global_idx]
            if track["source_type"] != local_type:
                continue
            if int(track["G_sem"]) != local_group:
                continue
            if not _labels_compatible(track["L_sem"], local_label):
                continue
            priority = _compute_local_track_seed_priority(mo, j, overlap)
            prev_local = best_seeded_local.get(seeded_global_idx)
            if (
                prev_local is None
                or priority > best_seeded_priority[seeded_global_idx]
            ):
                if prev_local is not None:
                    rejected_seeded_locals.add(int(prev_local))
                best_seeded_local[seeded_global_idx] = int(j)
                best_seeded_priority[seeded_global_idx] = priority
                rejected_seeded_locals.discard(int(j))
            else:
                rejected_seeded_locals.add(int(j))

        for seeded_global_idx, j in best_seeded_local.items():
            local_to_global[int(j)] = int(seeded_global_idx)
            seeded_locals.add(int(j))

    if chunk_idx > 0 and overlap > 0 and global_tracks:
        candidates: List[Tuple[float, int, int]] = []
        for j in range(mo.num_masklets):
            if j in seeded_locals or j in rejected_seeded_locals:
                continue
            for g, track in enumerate(global_tracks):
                if g in local_to_global.values():
                    continue
                score = _compute_track_match_score(mo, j, track, start, overlap)
                if track["source_type"] == "thing_tracked":
                    threshold = 0.28
                elif track["source_type"] == "structure_tracked":
                    threshold = 0.16
                elif track["G_sem"] == 0:
                    threshold = 0.18
                else:
                    threshold = 0.30
                if score >= threshold:
                    candidates.append((score, j, g))
        candidates.sort(key=lambda x: x[0], reverse=True)
        used_local = set(seeded_locals)
        used_global = set(local_to_global.values())
        for score, j, g in candidates:
            if j in used_local or g in used_global:
                continue
            local_to_global[j] = g
            used_local.add(j)
            used_global.add(g)

    preserve_overlap_until = start + overlap
    for j in range(mo.num_masklets):
        if j in rejected_seeded_locals and j not in local_to_global:
            continue
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
    consolidate_primary_person: bool = False,
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

    finalize_t0 = time.time()
    print(
        f"Finalizing global tracks: start with {len(global_tracks)} sparse tracks",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, thing_structure_dedup_debug = _deduplicate_thing_structure_tracks(
        global_tracks, total_frames,
    )
    print(
        f"Finalizing global tracks: thing/structure dedup -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, structure_dedup_debug = _deduplicate_structure_stuff_tracks(
        global_tracks, total_frames,
    )
    print(
        f"Finalizing global tracks: structure/stuff dedup -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, mask_clean_debug = _clean_sparse_track_masks(global_tracks)
    print(
        f"Finalizing global tracks: mask clean -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, structure_regularize_debug = _regularize_structure_track_masks(global_tracks)
    print(
        f"Finalizing global tracks: structure regularize -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, short_gap_bridge_debug = _bridge_short_track_gaps(
        global_tracks, total_frames,
    )
    print(
        f"Finalizing global tracks: short gap bridge -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, frame_suppress_debug = _suppress_duplicate_track_frames(
        global_tracks, total_frames,
    )
    print(
        f"Finalizing global tracks: frame suppress -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, redundant_drop_debug = _drop_redundant_global_tracks(global_tracks)
    print(
        f"Finalizing global tracks: redundant drop -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    if consolidate_primary_person:
        global_tracks, primary_person_consolidate_debug = _consolidate_primary_person_tracks(
            global_tracks,
            total_frames,
        )
    else:
        primary_person_consolidate_debug = []
    print(
        f"Finalizing global tracks: primary person consolidate -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, person_false_drop_debug = _drop_small_person_false_tracks(
        global_tracks,
        total_frames,
    )
    print(
        f"Finalizing global tracks: person false-track drop -> {len(global_tracks)} "
        f"tracks in {time.time() - step_t0:.2f}s",
        flush=True,
    )
    step_t0 = time.time()
    global_tracks, prune_debug = _prune_weak_global_tracks(global_tracks)
    print(
        f"Finalizing global tracks: prune -> {len(global_tracks)} tracks in "
        f"{time.time() - step_t0:.2f}s; total finalize {time.time() - finalize_t0:.2f}s",
        flush=True,
    )
    return SparseMaskletOutput(
        tracks=global_tracks,
        num_masklets=len(global_tracks),
        num_frames=total_frames,
        frame_height=H,
        frame_width=W,
        debug={
            "chunk_match_debug": chunk_match_debug,
            "thing_structure_dedup_debug": thing_structure_dedup_debug,
            "structure_dedup_debug": structure_dedup_debug,
            "mask_clean_debug": mask_clean_debug,
            "structure_regularize_debug": structure_regularize_debug,
            "short_gap_bridge_debug": short_gap_bridge_debug,
            "frame_suppress_debug": frame_suppress_debug,
            "redundant_drop_debug": redundant_drop_debug,
            "primary_person_consolidate_debug": primary_person_consolidate_debug,
            "person_false_drop_debug": person_false_drop_debug,
            "prune_debug": prune_debug,
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

    def _render_rank(track_idx: int) -> int:
        if isinstance(mo, SparseMaskletOutput):
            source_type = str(mo.tracks[track_idx].get("source_type", "?"))
        else:
            source_type = mo.source_type[track_idx] if track_idx < len(mo.source_type) else "?"
        if source_type in {"stuff_static", "structure_tracked"}:
            return 0
        return 1

    for j in sorted(range(J), key=_render_rank):
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
            source_type = str(track.get("source_type", "?"))
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
            source_type = mo.source_type[j] if j < len(mo.source_type) else "?"

        colour = get_colour(j)
        colour_np = np.array(colour, dtype=np.uint8)
        local_alpha = mask_alpha * 0.55 if source_type in {"stuff_static", "structure_tracked"} else mask_alpha

        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1 - local_alpha)
            + colour_np.astype(np.float32) * local_alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, colour, 1 if source_type == "stuff_static" else 2)

        x1, y1, x2, y2 = box
        if source_type == "stuff_static":
            continue

        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        text = f"#{j} {label} [{group_name}] w={w_sem:.2f} q={q:.2f}"

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
    apply_sam31_route_preset(args, sys.argv[1:])

    if bool(args.efficientsam3_stuff_worker):
        _run_efficientsam3_stuff_worker(args)
        return
    if bool(args.clipseg_stuff_worker):
        _run_clipseg_stuff_worker(args)
        return
    if bool(args.groupvit_stuff_worker):
        _run_groupvit_stuff_worker(args)
        return
    if bool(args.lseg_stuff_worker):
        _run_lseg_stuff_worker(args)
        return

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

    temp_dirs: List[str] = []
    try:
        image_paths, temp_dir = collect_image_paths(
            args.input, args.start_frame, args.end_frame, args.stride,
        )
        if temp_dir:
            temp_dirs.append(temp_dir)
        if not image_paths:
            sys.exit("No images found.")
        print(f"Collected {len(image_paths)} images.")
        original_image_paths = list(image_paths)
        total_frames = int(len(image_paths))
        processing_max_side = int(args.processing_max_side)
        image_paths, resize_temp_dir, (orig_h, orig_w), (sample_h, sample_w) = (
            prepare_processing_image_paths(image_paths, processing_max_side)
        )
        if resize_temp_dir:
            temp_dirs.append(resize_temp_dir)
            print(
                f"Processing resize: original=({orig_h}, {orig_w}) "
                f"-> inference=({sample_h}, {sample_w}) max_side={processing_max_side}"
            )
        else:
            print(f"Processing resize: disabled/unchanged, frame=({sample_h}, {sample_w})")
        print(f"Frame size: ({sample_h}, {sample_w})  total_frames={total_frames}")

        detector_image_paths = image_paths
        detector_max_side = int(args.detector_max_side)
        if detector_max_side > 0:
            detector_image_paths, detector_resize_temp_dir, _, (det_h, det_w) = (
                prepare_processing_image_paths(original_image_paths, detector_max_side)
            )
            if detector_resize_temp_dir:
                temp_dirs.append(detector_resize_temp_dir)
            print(
                f"Detector resize: original=({orig_h}, {orig_w}) "
                f"-> detector=({det_h}, {det_w}) max_side={detector_max_side}"
            )
        else:
            print(f"Detector resize: reusing tracking frames ({sample_h}, {sample_w})")

        effective_chunk_size = int(args.chunk_size)
        if (
            args.sam_backend == "sam31_multiplex"
            and effective_chunk_size > 0
            and int(args.sam31_min_chunk_size) > 0
        ):
            effective_chunk_size = max(effective_chunk_size, int(args.sam31_min_chunk_size))

        chunks = split_into_chunks(total_frames, effective_chunk_size, args.chunk_overlap)
        chunk_size_note = str(effective_chunk_size or total_frames)
        if effective_chunk_size != int(args.chunk_size):
            chunk_size_note += f" (requested {args.chunk_size})"
        print(
            f"Chunk schedule: {len(chunks)} chunk(s)  "
            f"size={chunk_size_note}  overlap={args.chunk_overlap}"
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
        elif bool(args.disable_stuff_prompts):
            frontend_kwargs["stuff_prompts"] = []

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
            sam31_offload_sam_during_detection=args.sam31_offload_sam_during_detection,
            sam31_enable_backward=bool(args.sam31_enable_backward),
            sam31_text_track_labels=args.sam31_text_track_labels,
            sam31_direct_text_prompt_labels=args.sam31_direct_text_prompt_labels,
            sam31_direct_text_prompt_frame_count=args.sam31_direct_text_prompt_frame_count,
            sam31_structure_prompt_labels=args.sam31_structure_prompt_labels,
            sam31_structure_prompt_frame_count=args.sam31_structure_prompt_frame_count,
            sam31_structure_prompt_chunk_stride=args.sam31_structure_prompt_chunk_stride,
            sam31_person_refresh_prompt_frames=args.sam31_person_refresh_prompt_frames,
            sam31_nontext_object_prompt_budget=args.sam31_nontext_object_prompt_budget,
            sam31_nontext_object_prompt_min_support=args.sam31_nontext_object_prompt_min_support,
            sam31_text_object_prompt_budget=args.sam31_text_object_prompt_budget,
            sam31_nontext_sparse_support=bool(args.sam31_nontext_sparse_support),
            sam31_max_text_prompt_objects=args.sam31_max_text_prompt_objects,
            sam31_max_internal_objects=args.sam31_max_internal_objects,
            device=args.device,
            detector_type=args.detector,
        )
        if args.detector == "gdino":
            build_kwargs["gdino_config"] = args.gdino_config
            build_kwargs["gdino_checkpoint"] = args.gdino_checkpoint
        elif args.detector == "yoloe":
            build_kwargs["yoloe_model"] = args.yoloe_model
            build_kwargs["yoloe_batch_size"] = args.yoloe_batch_size
            build_kwargs["yoloe_imgsz"] = args.yoloe_imgsz

        frontend = VideoMaskletFrontend.from_config(**build_kwargs, **frontend_kwargs)
        print(f"Models loaded in {time.time() - t0:.1f}s")

        # Run Stage C chunk-by-chunk
        print("Running Video Masklet Front-end (Stage C) ...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.time()

        global_tracks: List[Dict] = []
        chunk_match_debug: List[Dict[str, object]] = []
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
                global_tracks
                and args.chunk_overlap > 0
            ):
                seed_detections_by_frame = _build_chunk_seed_detections(
                    global_tracks,
                    start,
                    args.chunk_overlap,
                    seed_carry_gap=args.seed_carry_gap,
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
                detector_image_paths=detector_image_paths[start:end],
            )
            local_sparse_with_stuff: Optional[SparseMaskletOutput] = None
            if bool(args.efficientsam3_stuff_enable) and bool(args.efficientsam3_stuff_chunk_mode):
                stuff_t0 = time.time()
                print(
                    "  Stage C chunk STUFF: running "
                    f"{str(args.stuff_backend).strip().lower()} on frames [{start}, {end}) ...",
                    flush=True,
                )
                stuff_backend = str(args.stuff_backend).strip().lower()
                _trim_cuda_cache()
                if stuff_backend == "clipseg":
                    if bool(args.clipseg_stuff_inprocess):
                        stuff_payload = _run_clipseg_stuff_inprocess_payload(
                            image_paths[start:end],
                            args,
                        )
                    else:
                        stuff_payload = _run_clipseg_stuff_subprocess_payload(
                            image_paths[start:end],
                            args,
                            temp_dirs,
                            prefix=f"loger_clipseg_stuff_chunk{ci:04d}_",
                        )
                elif stuff_backend == "groupvit":
                    stuff_payload = _run_groupvit_stuff_subprocess_payload(
                        image_paths[start:end],
                        args,
                        temp_dirs,
                        prefix=f"loger_groupvit_stuff_chunk{ci:04d}_",
                    )
                elif stuff_backend == "lseg":
                    if bool(args.lseg_stuff_inprocess):
                        stuff_payload = _run_lseg_stuff_inprocess_payload(
                            image_paths[start:end],
                            args,
                        )
                    else:
                        stuff_payload = _run_lseg_stuff_subprocess_payload(
                            image_paths[start:end],
                            args,
                            temp_dirs,
                            prefix=f"loger_lseg_stuff_chunk{ci:04d}_",
                        )
                else:
                    stuff_payload = _run_efficientsam3_stuff_subprocess_payload(
                        image_paths[start:end],
                        args,
                        temp_dirs,
                        prefix=f"loger_efficientsam3_stuff_chunk{ci:04d}_",
                    )
                _trim_cuda_cache()
                local_sparse = _masklet_output_to_sparse_local(chunk_output)
                local_sparse_with_stuff = _merge_efficientsam3_stuff_tracks(
                    local_sparse,
                    stuff_payload,
                    replace_existing=bool(args.efficientsam3_stuff_replace_existing),
                    subtract_things=bool(args.efficientsam3_stuff_subtract_things),
                    subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
                )
                stuff_debug = stuff_payload.get("debug", {})
                print(
                    "  Stage C chunk STUFF done in "
                    f"{time.time() - stuff_t0:.2f}s: "
                    f"+{stuff_debug.get('efficientsam3_stuff_tracks_added', stuff_debug.get('clipseg_stuff_tracks_added', stuff_debug.get('groupvit_stuff_tracks_added', stuff_debug.get('lseg_stuff_tracks_added', 0))))} tracks, "
                    f"+{stuff_debug.get('efficientsam3_stuff_masks_added', stuff_debug.get('clipseg_stuff_masks_added', stuff_debug.get('groupvit_stuff_masks_added', stuff_debug.get('lseg_stuff_masks_added', 0))))} masks",
                    flush=True,
                )
            print(f"  Stage C chunk done in {time.time() - chunk_t0:.2f}s")
            if chunk_output.debug.get("discovery_frames") is not None:
                print(f"  Discovery frames (local): {chunk_output.debug['discovery_frames']}")
            print_masklet_output(
                local_sparse_with_stuff
                if local_sparse_with_stuff is not None
                else chunk_output
            )
            chunk_debug = merge_chunk_masklet_into_global(
                global_tracks,
                chunk_output,
                chunk_idx=ci,
                start=start,
                end=end,
                total_frames=total_frames,
                chunk_overlap=args.chunk_overlap,
            )
            if local_sparse_with_stuff is not None:
                stuff_sparse = SparseMaskletOutput(
                    tracks=[
                        track for track in local_sparse_with_stuff.tracks
                        if str(track.get("source_type", "")) == "stuff_static"
                    ],
                    num_masklets=sum(
                        1 for track in local_sparse_with_stuff.tracks
                        if str(track.get("source_type", "")) == "stuff_static"
                    ),
                    num_frames=local_sparse_with_stuff.num_frames,
                    frame_height=local_sparse_with_stuff.frame_height,
                    frame_width=local_sparse_with_stuff.frame_width,
                    debug=dict(local_sparse_with_stuff.debug),
                )
                chunk_debug["chunk_stuff_merge_debug"] = _merge_chunk_sparse_tracks_into_global(
                    global_tracks,
                    stuff_sparse,
                    chunk_idx=ci,
                    start=start,
                    end=end,
                    total_frames=total_frames,
                    chunk_overlap=args.chunk_overlap,
                    source_types={"stuff_static"},
                )
            chunk_match_debug.append(chunk_debug)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        masklet_output = finalize_global_tracks(
            global_tracks,
            total_frames=total_frames,
            H=sample_h,
            W=sample_w,
            chunk_match_debug=chunk_match_debug,
            consolidate_primary_person=bool(args.consolidate_primary_person),
        )
        try:
            if hasattr(frontend, "detector") and hasattr(frontend.detector, "release_gpu"):
                frontend.detector.release_gpu()
        except Exception:
            pass
        del frontend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if (sample_h, sample_w) != (orig_h, orig_w):
            print(
                f"Upscaling sparse masklets back to original frame size "
                f"({orig_h}, {orig_w})",
                flush=True,
            )
            masklet_output = upscale_sparse_masklet_output(masklet_output, orig_h, orig_w)

        if bool(args.efficientsam3_stuff_enable) and not bool(args.efficientsam3_stuff_chunk_mode):
            stuff_backend = str(args.stuff_backend).strip().lower()
            _trim_cuda_cache()
            if stuff_backend == "clipseg":
                if bool(args.clipseg_stuff_inprocess):
                    payload = _run_clipseg_stuff_inprocess_payload(original_image_paths, args)
                    masklet_output = _merge_efficientsam3_stuff_tracks(
                        masklet_output,
                        payload,
                        replace_existing=bool(args.efficientsam3_stuff_replace_existing),
                        subtract_things=bool(args.efficientsam3_stuff_subtract_things),
                        subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
                    )
                else:
                    masklet_output = _augment_with_clipseg_stuff_subprocess(
                        masklet_output,
                        original_image_paths,
                        args,
                        temp_dirs,
                    )
            elif stuff_backend == "groupvit":
                masklet_output = _augment_with_groupvit_stuff_subprocess(
                    masklet_output,
                    original_image_paths,
                    args,
                    temp_dirs,
                )
            elif stuff_backend == "lseg":
                if bool(args.lseg_stuff_inprocess):
                    payload = _run_lseg_stuff_inprocess_payload(original_image_paths, args)
                    masklet_output = _merge_efficientsam3_stuff_tracks(
                        masklet_output,
                        payload,
                        replace_existing=bool(args.efficientsam3_stuff_replace_existing),
                        subtract_things=bool(args.efficientsam3_stuff_subtract_things),
                        subtract_dilation=int(args.efficientsam3_stuff_subtract_dilation),
                    )
                else:
                    masklet_output = _augment_with_lseg_stuff_subprocess(
                        masklet_output,
                        original_image_paths,
                        args,
                        temp_dirs,
                    )
            else:
                masklet_output = _augment_with_efficientsam3_stuff_subprocess(
                    masklet_output,
                    original_image_paths,
                    args,
                    temp_dirs,
                )
            _trim_cuda_cache()

        _release_clipseg_stuff_cache()
        _release_lseg_stuff_cache()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stage_c_elapsed = time.time() - t0
        print(
            f"\nStage C total done in {stage_c_elapsed:.2f}s "
            f"({total_frames / max(stage_c_elapsed, 1e-6):.3f} fps)"
        )
        if torch.cuda.is_available():
            peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
            curr_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            curr_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(
                "CUDA memory Stage C: "
                f"peak_alloc={peak_alloc:.2f}GiB "
                f"peak_reserved={peak_reserved:.2f}GiB "
                f"current_alloc={curr_alloc:.2f}GiB "
                f"current_reserved={curr_reserved:.2f}GiB"
            )

        print_masklet_output(masklet_output)

        create_tracking_video(
            original_image_paths, masklet_output, args.output_video,
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
        for cleanup_dir in reversed(temp_dirs):
            if cleanup_dir and os.path.isdir(cleanup_dir):
                shutil.rmtree(cleanup_dir)


if __name__ == "__main__":
    main()
