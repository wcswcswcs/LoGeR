#!/usr/bin/env python3
"""
Stage A standalone runner — run LoGeR geometry backbone inference and
inspect / save / visualise the structured output.

Usage examples::

    # Minimal — print summary only
    python run_geometry_backbone_inference.py \
        --input data/examples/office \
        --config ckpts/LoGeR_star/original_config.yaml \
        --checkpoint ckpts/LoGeR_star/latest.pt

    # 3D interactive visualisation (viser, opens in browser)
    python run_geometry_backbone_inference.py \
        --input data/examples/office \
        --config ckpts/LoGeR_star/original_config.yaml \
        --checkpoint ckpts/LoGeR_star/latest.pt \
        --viser --share

    # Save 2D visualisation images (depth, confidence, …)
    python run_geometry_backbone_inference.py \
        --input data/examples/office \
        --config ckpts/LoGeR/original_config.yaml \
        --checkpoint ckpts/LoGeR/latest.pt \
        --save_vis results/vis_office

    # Save .pt tensor output
    python run_geometry_backbone_inference.py \
        --input data/examples/office \
        --config ckpts/LoGeR/original_config.yaml \
        --checkpoint ckpts/LoGeR/latest.pt \
        --start_frame 0 --end_frame 50 --stride 1 \
        --window_size 32 --overlap_size 3 \
        --resolution 504 280 \
        --output results/office_geometry.pt

    # From a video file
    python run_geometry_backbone_inference.py \
        --input /path/to/video.mp4 \
        --config ckpts/LoGeR_star/original_config.yaml \
        --checkpoint ckpts/LoGeR_star/latest.pt \
        --end_frame 100 --viser
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np
import torch
from natsort import natsorted

from loger.pipeline.geometry_backbone import (
    GeometryOutput,
    WriteCacheOutput,
    LoGeRGeometryBackbone,
    load_images,
)
from loger.utils.viser_utils import viser_wrapper


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run LoGeR Geometry Backbone (Stage A) and save structured output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input / output
    p.add_argument("--input", required=True, help="Image folder or video file.")
    p.add_argument("--output", default=None, help="Path to save the .pt output. If omitted, results are only printed.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1, help="-1 = all frames")
    p.add_argument("--stride", type=int, default=1)

    # Model
    p.add_argument("--checkpoint", required=True, help="Path to latest.pt")
    p.add_argument("--config", default=None, help="Path to original_config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Resolution
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="Target (W, H); must be multiples of 14.  Auto-computed if omitted.")

    # Inference
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap_size", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=0)
    p.add_argument("--se3", action="store_true", default=None,
                    help="Enable SE(3) alignment.  Auto-detected from config if omitted.")

    # TTT cache export
    p.add_argument("--cache_ttt", action="store_true",
                    help="Export WriteCacheOutput (TTT update primitives) for delayed write-back.")

    # Visualisation
    p.add_argument("--viser", action="store_true",
                    help="Launch interactive 3D viser viewer in the browser.")
    p.add_argument("--port", type=int, default=8080, help="Viser server port.")
    p.add_argument("--share", action="store_true", help="Create a public viser share link.")
    p.add_argument("--subsample", type=int, default=2,
                    help="Point cloud subsample factor for viser (lower = denser).")
    p.add_argument("--conf_threshold", type=float, default=20.0,
                    help="Initial confidence threshold (%%) for viser point cloud filtering.")
    p.add_argument("--save_vis", default=None, metavar="DIR",
                    help="Save 2D visualisation images (depth, confidence, …) to DIR.")
    p.add_argument("--vis_frames", type=int, nargs="*", default=None,
                    help="Frame indices to visualise with --save_vis (default: first, mid, last).")
    return p


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}


def is_video(path: str) -> bool:
    return os.path.isfile(path) and os.path.splitext(path)[1].lower() in VIDEO_EXTS


def extract_frames(video_path: str, out_dir: str, start: int, end: int, stride: int) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_end = total - 1 if end == -1 else min(end, total - 1)

    paths: list[str] = []
    idx = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or idx > actual_end:
            break
        if idx >= start and (idx - start) % stride == 0:
            p = os.path.join(out_dir, f"frame_{count:06d}.png")
            cv2.imwrite(p, frame)
            paths.append(p)
            count += 1
        idx += 1
    cap.release()
    print(f"Extracted {count} frames from {video_path}")
    return natsorted(paths)


def collect_image_paths(input_path: str, start: int, end: int, stride: int) -> tuple[list[str], str | None]:
    """Return (image_paths, optional_temp_dir)."""
    if is_video(input_path):
        tmp = tempfile.mkdtemp(prefix="loger_frames_")
        paths = extract_frames(input_path, tmp, start, end, stride)
        return paths, tmp

    if os.path.isdir(input_path):
        paths = natsorted(
            glob.glob(os.path.join(input_path, "*.png"))
            + glob.glob(os.path.join(input_path, "*.jpg"))
            + glob.glob(os.path.join(input_path, "*.jpeg"))
        )
        paths = [p for p in paths if "depth" not in os.path.basename(p).lower()]
        end_idx = None if end == -1 else end
        paths = paths[start:end_idx:stride]
        return paths, None

    raise FileNotFoundError(f"Input path does not exist or is not a recognised format: {input_path}")


# ---------------------------------------------------------------------------
# 2D visualisation helpers
# ---------------------------------------------------------------------------
def _colorize(tensor_2d: torch.Tensor, cmap_name: str = "turbo") -> np.ndarray:
    """Map a [H, W] float tensor to an [H, W, 3] uint8 colour image."""
    import matplotlib.cm as cm
    arr = tensor_2d.numpy()
    lo, hi = np.nanpercentile(arr[np.isfinite(arr)], [2, 98]) if np.isfinite(arr).any() else (0, 1)
    if hi - lo < 1e-8:
        hi = lo + 1
    normalised = np.clip((arr - lo) / (hi - lo), 0, 1)
    cmap = cm.get_cmap(cmap_name)
    coloured = (cmap(normalised)[..., :3] * 255).astype(np.uint8)
    return coloured


def save_2d_visualisations(
    geo: GeometryOutput,
    images: torch.Tensor,
    out_dir: str,
    frame_indices: list[int] | None = None,
) -> None:
    """Save per-frame depth, confidence and RGB overlay images.

    Parameters
    ----------
    geo : GeometryOutput
    images : [T, 3, H, W] tensor (values in [0,1])
    out_dir : directory to write into
    frame_indices : which frames to visualise; None = first, mid, last
    """
    os.makedirs(out_dir, exist_ok=True)
    T = geo.num_frames

    if frame_indices is None:
        frame_indices = sorted({0, T // 2, T - 1})
    frame_indices = [i for i in frame_indices if 0 <= i < T]

    for t in frame_indices:
        tag = f"frame_{t:04d}"

        # --- RGB ---
        rgb_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_rgb.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        # --- Depth (z channel of local_points) ---
        if geo.local_points is not None:
            depth = geo.local_points[t, :, :, 2]  # [H_p, W_p]
            depth_vis = _colorize(depth, "turbo")
            cv2.imwrite(os.path.join(out_dir, f"{tag}_depth.png"), cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

        # --- Confidence ---
        if geo.confidence is not None:
            conf = geo.confidence[t]  # [H_p, W_p]
            conf_vis = _colorize(conf, "magma")
            cv2.imwrite(os.path.join(out_dir, f"{tag}_confidence.png"), cv2.cvtColor(conf_vis, cv2.COLOR_RGB2BGR))

        # --- Confidence binary mask (>0.5 vs <=0.5) ---
        if geo.confidence is not None:
            mask = (geo.confidence[t] > 0.5).numpy().astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out_dir, f"{tag}_conf_mask.png"), mask)

    print(f"Saved 2D visualisations for {len(frame_indices)} frames to {out_dir}/")


# ---------------------------------------------------------------------------
# Viser 3D visualisation bridge
# ---------------------------------------------------------------------------
def launch_viser(
    geo: GeometryOutput,
    images: torch.Tensor,
    port: int = 8080,
    share: bool = False,
    subsample: int = 2,
    conf_threshold: float = 20.0,
) -> None:
    """Build the dict that ``viser_wrapper`` expects and launch the viewer."""
    pred_np: dict[str, np.ndarray] = {}

    if geo.world_points is not None:
        pred_np["points"] = geo.world_points.numpy()           # [T, H, W, 3]
    if geo.camera_poses is not None:
        pred_np["camera_poses"] = geo.camera_poses.numpy()     # [T, 4, 4]
    if geo.confidence is not None:
        conf = geo.confidence
        if conf.dim() == 3:
            conf = conf.unsqueeze(-1)                          # [T, H, W, 1]
        pred_np["conf"] = conf.numpy()

    # images: [T, 3, H, W] -> [T, H, W, 3]
    pred_np["images"] = images.permute(0, 2, 3, 1).numpy()

    print(f"Starting viser 3D viewer on port {port} …")
    viser_wrapper(
        pred_np,
        port=port,
        init_conf_threshold=conf_threshold,
        subsample=subsample,
        share=share,
    )


# ---------------------------------------------------------------------------
# Pretty-printing helper
# ---------------------------------------------------------------------------
def print_geometry_output(geo: GeometryOutput) -> None:
    print("\n" + "=" * 60)
    print("GeometryOutput summary")
    print("=" * 60)
    print(f"  num_frames           : {geo.num_frames}")
    print(f"  pointmap_resolution  : {geo.pointmap_resolution}  (H_p, W_p)")
    print(f"  patch_grid           : {geo.patch_grid}  (H_tok, W_tok)")

    def _shape(t):
        return tuple(t.shape) if t is not None else None

    print(f"  local_points.shape   : {_shape(geo.local_points)}")
    print(f"  world_points.shape   : {_shape(geo.world_points)}")
    print(f"  camera_poses.shape   : {_shape(geo.camera_poses)}")
    print(f"  confidence.shape     : {_shape(geo.confidence)}")
    print(f"  patch_meta.shape     : {_shape(geo.patch_meta)}")
    print(f"  token_type.shape     : {_shape(geo.token_type)}")

    L_tok = geo.token_type.shape[0] if geo.token_type is not None else 0
    if geo.token_type is not None:
        n_reg = (geo.token_type == 0).sum().item()
        n_role = (geo.token_type == 1).sum().item()
        n_patch = (geo.token_type == 2).sum().item()
        print(f"  token breakdown      : {n_reg} reg + {n_role} role + {n_patch} patch = {L_tok} total")

    if geo.confidence is not None:
        c = geo.confidence
        print(f"  confidence range     : [{c.min().item():.4f}, {c.max().item():.4f}]  mean={c.mean().item():.4f}")

    if geo.camera_poses is not None:
        t_norms = geo.camera_poses[:, :3, 3].norm(dim=-1)
        print(f"  camera translation ‖t‖: min={t_norms.min().item():.4f}  max={t_norms.max().item():.4f}")

    print(f"  raw_predictions keys : {sorted(geo.raw_predictions.keys())}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # Validate paths early
    if args.config and not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # ------------------------------------------------------------------
    # 1. Collect images
    # ------------------------------------------------------------------
    image_paths, temp_dir = collect_image_paths(
        args.input, args.start_frame, args.end_frame, args.stride
    )
    if not image_paths:
        sys.exit("No images found. Check --input path and frame range.")
    print(f"Collected {len(image_paths)} images.")

    # ------------------------------------------------------------------
    # 2. Load images into tensor
    # ------------------------------------------------------------------
    target_w, target_h = (args.resolution if args.resolution else (None, None))
    images = load_images(image_paths, target_w=target_w, target_h=target_h)
    print(f"Image tensor: {tuple(images.shape)}  (T, C, H, W)")

    # ------------------------------------------------------------------
    # 3. Build backbone
    # ------------------------------------------------------------------
    backbone_kwargs = dict(
        device=args.device,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
    )
    if args.se3 is not None:
        backbone_kwargs["se3"] = args.se3

    print("Loading LoGeR model …")
    t0 = time.time()
    backbone = LoGeRGeometryBackbone.from_config(
        checkpoint=args.checkpoint,
        config=args.config,
        **backbone_kwargs,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Run inference
    # ------------------------------------------------------------------
    print("Running geometry inference …")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    write_cache = None
    if args.cache_ttt:
        geo_output, write_cache = backbone.run(images, cache_ttt_primitives=True)
    else:
        geo_output = backbone.run(images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    fps = geo_output.num_frames / elapsed if elapsed > 0 else 0
    print(f"Inference done in {elapsed:.2f}s  ({fps:.1f} FPS)")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print_geometry_output(geo_output)

    if write_cache is not None:
        print(f"WriteCacheOutput: {write_cache.num_ttt_layers} TTT layer caches, "
              f"patch_grid={write_cache.patch_grid}, "
              f"num_frames={write_cache.num_frames}")

    # ------------------------------------------------------------------
    # 6. Optionally save .pt
    # ------------------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_dict = {
            "local_points": geo_output.local_points,
            "world_points": geo_output.world_points,
            "camera_poses": geo_output.camera_poses,
            "confidence": geo_output.confidence,
            "patch_meta": geo_output.patch_meta,
            "token_type": geo_output.token_type,
            "num_frames": geo_output.num_frames,
            "pointmap_resolution": geo_output.pointmap_resolution,
            "patch_grid": geo_output.patch_grid,
        }
        torch.save(save_dict, args.output)
        print(f"Saved geometry output to {args.output}")

    # ------------------------------------------------------------------
    # 7. 2D visualisation images
    # ------------------------------------------------------------------
    if args.save_vis:
        save_2d_visualisations(
            geo_output, images, args.save_vis, frame_indices=args.vis_frames,
        )

    # ------------------------------------------------------------------
    # 8. Interactive 3D viser viewer
    # ------------------------------------------------------------------
    if args.viser:
        launch_viser(
            geo_output,
            images,
            port=args.port,
            share=args.share,
            subsample=args.subsample,
            conf_threshold=args.conf_threshold,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
