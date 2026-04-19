#!/usr/bin/env python3
"""
Stage C standalone runner -- Video Masklet Front-end with visualised output.

Supports two detector back-ends:
  * ``gdino``  – Grounding DINO  + SAM 2.1 image refinement
  * ``yoloe``  – YOLOE (ultralytics) with built-in segmentation

Usage examples::

    # --- Grounding DINO ---
    CUDA_VISIBLE_DEVICES=1 conda run -n loger python run_video_masklet_front_end.py \\
        --input data/examples/office \\
        --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \\
        --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
        --detector gdino \\
        --gdino_config Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
        --gdino_checkpoint /mnt/data/users/chengshun.wang/pjs/GroundingDINO/weights/groundingdino_swint_ogc.pth \\
        --output_video results/office_masklets.mp4

    # --- YOLOE ---
    CUDA_VISIBLE_DEVICES=1 conda run -n loger python run_video_masklet_front_end.py \\
        --input data/examples/office \\
        --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \\
        --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
        --detector yoloe \\
        --yoloe_model yoloe-11l-seg.pt \\
        --output_video results/office_yoloe.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import Optional

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

    # SAM2
    p.add_argument("--sam2_checkpoint", required=True)
    p.add_argument("--sam2_model_cfg", required=True)

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
                   help="Annotation frame index (detector runs once on this frame).")
    p.add_argument("--max_thing_objects", type=int, default=15,
                   help="Max thing objects to track with SAM2.")

    # Video
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mask_alpha", type=float, default=0.40)

    return p


def load_images(paths: list) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    return torch.stack([to_tensor(Image.open(p).convert("RGB")) for p in paths])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def render_annotated_frame(
    rgb_np: np.ndarray,
    mo: MaskletOutput,
    frame_idx: int,
    mask_alpha: float = 0.40,
) -> np.ndarray:
    canvas = rgb_np.copy()
    overlay = canvas.copy()
    J = mo.num_masklets

    for j in range(J):
        if not mo.V_mask[j, frame_idx]:
            continue
        mask = mo.M_mask[j, frame_idx].numpy().astype(bool)
        if mask.sum() == 0:
            continue

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

        box = mo.B_mask[j, frame_idx].numpy().astype(int)
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, 2)

        label = mo.L_sem[j] if j < len(mo.L_sem) else "?"
        group_name = SEMANTIC_GROUP_NAMES.get(mo.G_sem[j].item(), "?")
        w_sem = mo.W_sem[j].item()
        q = mo.Q_mask[j, frame_idx].item()
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
    images: torch.Tensor,
    mo: MaskletOutput,
    output_path: str,
    fps: int = 10,
    mask_alpha: float = 0.40,
    save_frames_dir: Optional[str] = None,
) -> None:
    T = images.shape[0]
    H, W = mo.frame_height, mo.frame_width

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    for t in range(T):
        rgb_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
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
def print_masklet_output(mo: MaskletOutput) -> None:
    print("\n" + "=" * 72)
    print("MaskletOutput summary  (Stage C: Video Masklet Front-end)")
    print("=" * 72)
    print(f"  num_masklets         : {mo.num_masklets}")
    print(f"  num_frames           : {mo.num_frames}")
    print(f"  frame_size           : {mo.frame_height} x {mo.frame_width}")
    print(f"  M_mask.shape         : {tuple(mo.M_mask.shape)}")

    jt = mo.debug.get("J_thing", "?")
    js = mo.debug.get("J_stuff", "?")
    print(f"  thing / stuff        : {jt} / {js}")

    if mo.num_masklets > 0:
        print()
        print(f"  {'ID':>3s}  {'Type':14s}  {'Label':20s}  {'Group':22s}  {'W_sem':>5s}  "
              f"{'Birth':>5s}  {'Visible':>7s}  {'MeanQ':>5s}  {'MeanArea':>8s}")
        print("  " + "-" * 115)
        for j in range(mo.num_masklets):
            vis = mo.V_mask[j].sum().item()
            mq = mo.Q_mask[j, mo.V_mask[j]].mean().item() if vis > 0 else 0
            ma = mo.A_ratio[j, mo.V_mask[j]].mean().item() if vis > 0 else 0
            g = mo.G_sem[j].item()
            gn = SEMANTIC_GROUP_NAMES.get(g, "?")
            lbl = mo.L_sem[j] if j < len(mo.L_sem) else "?"
            bf = mo.birth_frame[j] if j < len(mo.birth_frame) else -1
            stype = mo.source_type[j] if j < len(mo.source_type) else "?"
            print(f"  {j:3d}  {stype:14s}  {lbl:20s}  {gn:22s}  {mo.W_sem[j].item():5.2f}  "
                  f"{bf:5d}  {vis:5.0f}/{mo.num_frames}  {mq:5.3f}  {ma:8.5f}")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    temp_dir = None
    try:
        image_paths, temp_dir = collect_image_paths_geo(
            args.input, args.start_frame, args.end_frame, args.stride,
        )
        if not image_paths:
            sys.exit("No images found.")
        print(f"Collected {len(image_paths)} images.")

        images = load_images(image_paths)
        print(f"Image tensor: {tuple(images.shape)}  (T, C, H, W)")

        # Build kwargs
        frontend_kwargs: dict = dict(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            ann_frame_idx=args.ann_frame_idx,
            max_thing_objects=args.max_thing_objects,
        )
        if args.thing_prompts:
            frontend_kwargs["thing_prompts"] = [s.strip() for s in args.thing_prompts.split(",")]
        if args.stuff_prompts:
            frontend_kwargs["stuff_prompts"] = [s.strip() for s in args.stuff_prompts.split(",")]

        # Build frontend
        print(f"Loading models (detector={args.detector}) ...")
        t0 = time.time()

        build_kwargs: dict = dict(
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
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

        # Run Stage C
        print("Running Video Masklet Front-end (Stage C) ...")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        masklet_output = frontend.run(images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"Stage C done in {time.time() - t0:.2f}s")

        print_masklet_output(masklet_output)

        create_tracking_video(
            images, masklet_output, args.output_video,
            fps=args.fps, mask_alpha=args.mask_alpha,
            save_frames_dir=args.save_frames,
        )

        if args.output_pt:
            os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
            torch.save({
                "M_mask": masklet_output.M_mask,
                "V_mask": masklet_output.V_mask,
                "B_mask": masklet_output.B_mask,
                "Q_mask": masklet_output.Q_mask,
                "L_sem": masklet_output.L_sem,
                "G_sem": masklet_output.G_sem,
                "W_sem": masklet_output.W_sem,
                "A_ratio": masklet_output.A_ratio,
                "num_masklets": masklet_output.num_masklets,
                "num_frames": masklet_output.num_frames,
            }, args.output_pt)
            print(f"Saved masklet tensors to {args.output_pt}")
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
