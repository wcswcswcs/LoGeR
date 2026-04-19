#!/usr/bin/env python3
"""
Stage A + B standalone runner -- run LoGeR geometry backbone followed by
the Dynamic Cue Extractor, then print detailed statistics for both outputs.

Usage examples::

    # Minimal -- print summary only
    python inference_dynamic_cue_extractor.py \\
        --input data/examples/office \\
        --config ckpts/LoGeR/original_config.yaml \\
        --checkpoint ckpts/LoGeR/latest.pt

    # Custom cue-extractor hyper-parameters
    python inference_dynamic_cue_extractor.py \\
        --input data/examples/office \\
        --config ckpts/LoGeR/original_config.yaml \\
        --checkpoint ckpts/LoGeR/latest.pt \\
        --k_intra 4 --sigma_pt 0.05

    # Save cue output to .pt
    python inference_dynamic_cue_extractor.py \\
        --input data/examples/office \\
        --config ckpts/LoGeR/original_config.yaml \\
        --checkpoint ckpts/LoGeR/latest.pt \\
        --output results/office_cues.pt

    # Save cue visualisation to video
    CUDA_VISIBLE_DEVICES=1 python inference_dynamic_cue_extractor.py \
        --input data/examples/taylor.mp4 \
        --config ckpts/LoGeR/original_config.yaml \
        --checkpoint ckpts/LoGeR/latest.pt \
        --end_frame 200 \
        --k_intra 4 \
        --chunk_size 32\
        --output_video results/taylor_cues_k_intra_4.mp4
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from run_geometry_backbone_inference import (
    collect_image_paths,
    print_geometry_output,
)
from loger.pipeline.geometry_backbone import (
    GeometryOutput,
    LoGeRGeometryBackbone,
    load_images,
)
from loger.pipeline.dynamic_cue_extractor import (
    CueOutput,
    DynamicCueExtractor,
    NUM_CUE_CHANNELS,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run Stage A (Geometry Backbone) + Stage B (Dynamic Cue Extractor).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # -- Input / output ------------------------------------------------------
    p.add_argument("--input", required=True, help="Image folder or video file.")
    p.add_argument("--output", default=None,
                   help="Path to save .pt cue output.  If omitted, results are only printed.")
    p.add_argument("--output_video", default=None,
                   help="Path to save cue visualisation video.")
    p.add_argument("--save_frames", default=None,
                   help="Optional directory to save visualised cue frames.")
    p.add_argument("--fps", type=int, default=10, help="Output video FPS.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1, help="-1 = all frames")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--chunk_size", type=int, default=0,
                   help="Frames per chunk (0 = all frames as one chunk).")
    p.add_argument("--chunk_overlap", type=int, default=0,
                   help="Overlap between consecutive chunks.")

    # -- Model ---------------------------------------------------------------
    p.add_argument("--checkpoint", required=True, help="Path to latest.pt")
    p.add_argument("--config", default=None, help="Path to original_config.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # -- Resolution ----------------------------------------------------------
    p.add_argument("--resolution", type=int, nargs=2, default=None,
                   metavar=("W", "H"),
                   help="Target (W, H); must be multiples of 14.")

    # -- Geometry backbone ---------------------------------------------------
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap_size", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=0)
    p.add_argument("--se3", action="store_true", default=None)

    # -- Dynamic Cue Extractor -----------------------------------------------
    p.add_argument("--k_intra", type=int, default=10,
                   help="Number of intra-chunk support frames per frame.")
    p.add_argument("--sigma_pt", type=float, default=0.25,
                   help="Scale for point-residual -> consistency kernel.")
    p.add_argument("--tau_occ", type=float, default=0.05,
                   help="Depth-difference threshold for occlusion detection.")
    p.add_argument("--alpha_1", type=float, default=0.8,
                   help="C_dyn: weight on (1 - C_stat).")
    p.add_argument("--alpha_3", type=float, default=0.5,
                   help="C_dyn: subtract weight on C_occ.")
    p.add_argument("--lambda_s", type=float, default=1.0,
                   help="G_write_geo: C_stat weight.")
    p.add_argument("--lambda_a", type=float, default=0.5,
                   help="G_write_geo: C_anchor weight.")
    p.add_argument("--lambda_d", type=float, default=0.8,
                   help="G_write_geo: C_dyn weight (subtracted).")
    p.add_argument("--lambda_o", type=float, default=0.3,
                   help="G_write_geo: C_occ weight (subtracted).")
    p.add_argument("--lambda_u", type=float, default=0.5,
                   help="G_write_geo: C_unc weight (subtracted).")

    return p


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def _stat_line(name: str, t: torch.Tensor) -> str:
    return (
        f"  {name:20s}: "
        f"min={t.min().item():8.4f}  "
        f"max={t.max().item():8.4f}  "
        f"mean={t.mean().item():8.4f}  "
        f"std={t.std().item():8.4f}"
    )


def print_cue_output(cue: CueOutput) -> None:
    print("\n" + "=" * 72)
    print("CueOutput summary  (Stage B: Dynamic Cue Extractor)")
    print("=" * 72)
    print(f"  num_frames           : {cue.num_frames}")
    print(f"  spatial_resolution   : {cue.spatial_resolution}  (H_p, W_p)")
    print(f"  patch_grid           : {cue.patch_grid}  (H_tok, W_tok)")
    print(f"  E_cue.shape          : {tuple(cue.E_cue.shape)}")
    print(f"  G_write_geo.shape    : {tuple(cue.G_write_geo.shape)}")
    if cue.E_cue_patch is not None:
        print(f"  E_cue_patch.shape    : {tuple(cue.E_cue_patch.shape)}")
    if cue.G_write_geo_patch is not None:
        print(f"  G_write_geo_patch.shape: {tuple(cue.G_write_geo_patch.shape)}")

    print()
    print("  Per-channel pixel-level statistics:")
    channel_names = ["C_stat", "C_dyn", "C_occ", "C_unc", "C_anchor"]
    for i, name in enumerate(channel_names):
        print(_stat_line(name, cue.E_cue[..., i]))
    print(_stat_line("G_write_geo", cue.G_write_geo))

    if cue.E_cue_patch is not None:
        print()
        print("  Per-channel patch-level statistics:")
        for i, name in enumerate(channel_names):
            print(_stat_line(name + " (patch)", cue.E_cue_patch[..., i]))
        if cue.G_write_geo_patch is not None:
            print(_stat_line("G_write_geo (patch)", cue.G_write_geo_patch))

    if cue.debug:
        print()
        print("  Debug info:")
        rpt = cue.debug.get("mean_point_residual")
        if rpt is not None:
            print(f"    mean_point_residual : {rpt:.6f}")
        sc = cue.debug.get("support_count_per_frame")
        if sc is not None:
            print(f"    support counts      : min={sc.min().item():.0f}  max={sc.max().item():.0f}")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------
def split_into_chunks(
    total_frames: int, chunk_size: int, overlap: int = 0,
) -> List[Tuple[int, int]]:
    """Return (start, end) index pairs for each chunk."""
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [(0, total_frames)]
    chunks = []
    step = max(chunk_size - overlap, 1)
    for s in range(0, total_frames, step):
        e = min(s + chunk_size, total_frames)
        chunks.append((s, e))
        if e == total_frames:
            break
    return chunks


def merge_chunk_cues(
    chunk_cues: List[CueOutput],
    chunk_overlap: int,
) -> CueOutput:
    """Merge per-chunk CueOutput into one sequence-level CueOutput."""
    if not chunk_cues:
        raise ValueError("chunk_cues must not be empty")

    e_cue_parts = []
    g_write_parts = []
    e_cue_patch_parts = []
    g_write_patch_parts = []
    debug = {"chunk_debug": []}

    use_patch = all(c.E_cue_patch is not None for c in chunk_cues)
    use_patch = use_patch and all(c.G_write_geo_patch is not None for c in chunk_cues)

    for i, c in enumerate(chunk_cues):
        drop = 0 if i == 0 else min(chunk_overlap, c.num_frames)
        if drop >= c.num_frames:
            continue
        e_cue_parts.append(c.E_cue[drop:])
        g_write_parts.append(c.G_write_geo[drop:])
        if use_patch:
            e_cue_patch_parts.append(c.E_cue_patch[drop:])
            g_write_patch_parts.append(c.G_write_geo_patch[drop:])
        debug["chunk_debug"].append(c.debug)

    if not e_cue_parts:
        e_cue_parts = [chunk_cues[0].E_cue]
        g_write_parts = [chunk_cues[0].G_write_geo]
        if use_patch:
            e_cue_patch_parts = [chunk_cues[0].E_cue_patch]
            g_write_patch_parts = [chunk_cues[0].G_write_geo_patch]

    E_cue = torch.cat(e_cue_parts, dim=0)
    G_write_geo = torch.cat(g_write_parts, dim=0)
    E_cue_patch = torch.cat(e_cue_patch_parts, dim=0) if use_patch else None
    G_write_geo_patch = torch.cat(g_write_patch_parts, dim=0) if use_patch else None

    merged = CueOutput(
        E_cue=E_cue,
        G_write_geo=G_write_geo,
        E_cue_patch=E_cue_patch,
        G_write_geo_patch=G_write_geo_patch,
        num_frames=E_cue.shape[0],
        spatial_resolution=chunk_cues[0].spatial_resolution,
        patch_grid=chunk_cues[0].patch_grid,
        debug=debug,
    )
    return merged


# ---------------------------------------------------------------------------
# Cue visualisation
# ---------------------------------------------------------------------------
def _colorize_map(
    map_2d: np.ndarray,
    cmap: int = cv2.COLORMAP_TURBO,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    map_2d = np.nan_to_num(map_2d, nan=0.0, posinf=1.0, neginf=0.0)
    if value_range is None:
        lo = float(np.percentile(map_2d, 2))
        hi = float(np.percentile(map_2d, 98))
    else:
        lo, hi = value_range
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((map_2d - lo) / (hi - lo), 0.0, 1.0)
    vis_u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis_u8, cmap)


def _render_panel(title: str, img_bgr: np.ndarray) -> np.ndarray:
    panel = img_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        panel, title, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def create_cue_video(
    images: torch.Tensor,
    cue: CueOutput,
    output_path: str,
    fps: int = 10,
    save_frames_dir: str | None = None,
) -> None:
    """Create a 2x4 panel video for cue channels and write-gate map."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    T, _, H, W = images.shape
    cue_maps = cue.E_cue.permute(0, 3, 1, 2).cpu()  # [T, 5, Hc, Wc]
    cue_maps = F.interpolate(cue_maps, size=(H, W), mode="bilinear", align_corners=False)
    cue_maps_np = cue_maps.numpy()

    g_write = cue.G_write_geo.unsqueeze(1).cpu()  # [T, 1, Hc, Wc]
    g_write = F.interpolate(g_write, size=(H, W), mode="bilinear", align_corners=False)
    g_write_np = g_write[:, 0].numpy()

    gap = 4
    grid_w = W * 4 + gap * 3
    grid_h = H * 2 + gap
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w, grid_h),
    )

    channel_names = ["C_stat", "C_dyn", "C_occ", "C_unc", "C_anchor"]
    for t in range(T):
        rgb = (images[t].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        fixed_range = (0.0, 1.0)
        c_stat = _render_panel(
            channel_names[0],
            _colorize_map(cue_maps_np[t, 0], cv2.COLORMAP_VIRIDIS, value_range=fixed_range),
        )
        c_dyn = _render_panel(
            channel_names[1],
            _colorize_map(cue_maps_np[t, 1], cv2.COLORMAP_TURBO, value_range=fixed_range),
        )
        c_occ = _render_panel(
            channel_names[2],
            _colorize_map(cue_maps_np[t, 2], cv2.COLORMAP_INFERNO, value_range=fixed_range),
        )
        c_unc = _render_panel(
            channel_names[3],
            _colorize_map(cue_maps_np[t, 3], cv2.COLORMAP_MAGMA, value_range=fixed_range),
        )
        c_anchor = _render_panel(
            channel_names[4],
            _colorize_map(cue_maps_np[t, 4], cv2.COLORMAP_PLASMA, value_range=fixed_range),
        )
        g_write_vis = _render_panel(
            "G_write_geo",
            _colorize_map(g_write_np[t], cv2.COLORMAP_OCEAN, value_range=fixed_range),
        )

        dyn_overlay = cv2.addWeighted(
            rgb_bgr, 0.45,
            _colorize_map(cue_maps_np[t, 1], cv2.COLORMAP_TURBO, value_range=fixed_range),
            0.55, 0,
        )
        dyn_overlay = _render_panel("C_dyn overlay", dyn_overlay)
        rgb_panel = _render_panel(f"RGB frame {t}", rgb_bgr)

        top = np.concatenate([
            rgb_panel, np.zeros((H, gap, 3), np.uint8),
            c_stat, np.zeros((H, gap, 3), np.uint8),
            c_dyn, np.zeros((H, gap, 3), np.uint8),
            c_occ,
        ], axis=1)
        bottom = np.concatenate([
            c_unc, np.zeros((H, gap, 3), np.uint8),
            c_anchor, np.zeros((H, gap, 3), np.uint8),
            g_write_vis, np.zeros((H, gap, 3), np.uint8),
            dyn_overlay,
        ], axis=1)
        frame = np.concatenate([top, np.zeros((gap, grid_w, 3), np.uint8), bottom], axis=0)

        writer.write(frame)
        if save_frames_dir:
            cv2.imwrite(os.path.join(save_frames_dir, f"frame_{t:05d}.jpg"), frame)

    writer.release()
    print(f"Saved cue visualisation video to {output_path}  ({T} frames, {fps} FPS)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # -- validate paths ------------------------------------------------------
    if args.config and not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    temp_dir = None
    try:
        # -------------------------------------------------------------------
        # 1.  Collect images
        # -------------------------------------------------------------------
        image_paths, temp_dir = collect_image_paths(
            args.input, args.start_frame, args.end_frame, args.stride,
        )
        if not image_paths:
            sys.exit("No images found.  Check --input path and frame range.")
        print(f"Collected {len(image_paths)} images.")

        # -------------------------------------------------------------------
        # 2.  Load images
        # -------------------------------------------------------------------
        target_w, target_h = args.resolution if args.resolution else (None, None)
        images = load_images(image_paths, target_w=target_w, target_h=target_h)
        print(f"Image tensor: {tuple(images.shape)}  (T, C, H, W)")

        # -------------------------------------------------------------------
        # 3.  Chunk schedule
        # -------------------------------------------------------------------
        total_frames = images.shape[0]
        chunks = split_into_chunks(total_frames, args.chunk_size, args.chunk_overlap)
        print(
            f"\nChunk schedule: {len(chunks)} chunk(s), "
            f"size={args.chunk_size or total_frames}, overlap={args.chunk_overlap}"
        )
        for ci, (s, e) in enumerate(chunks):
            print(f"  chunk {ci}: frames [{s}, {e})")

        # -------------------------------------------------------------------
        # 4.  Build geometry backbone + extractor
        # -------------------------------------------------------------------
        backbone_kwargs: dict = dict(
            device=args.device,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            reset_every=args.reset_every,
            update_ttt_weights=True,
        )
        if args.se3 is not None:
            backbone_kwargs["se3"] = args.se3

        print("\nLoading LoGeR model ...")
        t0 = time.time()
        backbone = LoGeRGeometryBackbone.from_config(
            checkpoint=args.checkpoint,
            config=args.config,
            **backbone_kwargs,
        )
        print(f"Model loaded in {time.time() - t0:.1f}s")

        cue_kwargs = dict(
            k_intra=args.k_intra,
            sigma_pt=args.sigma_pt,
            tau_occ=args.tau_occ,
            alpha_1=args.alpha_1,
            alpha_3=args.alpha_3,
            lambda_s=args.lambda_s,
            lambda_a=args.lambda_a,
            lambda_d=args.lambda_d,
            lambda_o=args.lambda_o,
            lambda_u=args.lambda_u,
        )
        extractor = DynamicCueExtractor(**cue_kwargs)

        # -------------------------------------------------------------------
        # 5.  Chunk-by-chunk Stage A + B
        # -------------------------------------------------------------------
        all_cues: List[CueOutput] = []
        merged_image_parts: List[torch.Tensor] = []

        for ci, (start, end) in enumerate(chunks):
            print(f"\n{'#' * 72}")
            print(f"# Chunk {ci}/{len(chunks)-1}  frames [{start}, {end})")
            print(f"{'#' * 72}")
            chunk_images = images[start:end]

            # Stage A
            print("  Stage A: Geometry Backbone ...")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            geo: GeometryOutput = backbone.run(chunk_images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_a = time.time() - t0
            fps_a = geo.num_frames / elapsed_a if elapsed_a > 0 else 0
            print(f"    done in {elapsed_a:.2f}s  ({fps_a:.1f} FPS)")
            print_geometry_output(geo)

            # Stage B
            print("  Stage B: Dynamic Cue Extractor ...")
            t0 = time.time()
            cue_chunk: CueOutput = extractor.run(geo)
            elapsed_b = time.time() - t0
            print(f"    done in {elapsed_b:.2f}s")
            print_cue_output(cue_chunk)
            all_cues.append(cue_chunk)

            drop = 0 if ci == 0 else min(args.chunk_overlap, chunk_images.shape[0])
            if drop < chunk_images.shape[0]:
                merged_image_parts.append(chunk_images[drop:])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cue = merge_chunk_cues(all_cues, args.chunk_overlap)
        images_merged = torch.cat(merged_image_parts, dim=0) if merged_image_parts else images
        print("\nMerged chunk outputs:")
        print_cue_output(cue)

        # -------------------------------------------------------------------
        # 6.  Optionally save visualisation video
        # -------------------------------------------------------------------
        if args.output_video:
            create_cue_video(
                images=images_merged,
                cue=cue,
                output_path=args.output_video,
                fps=args.fps,
                save_frames_dir=args.save_frames,
            )

        # -------------------------------------------------------------------
        # 7.  Optionally save .pt
        # -------------------------------------------------------------------
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            save_dict = {
                "E_cue": cue.E_cue,
                "G_write_geo": cue.G_write_geo,
                "num_frames": cue.num_frames,
                "spatial_resolution": cue.spatial_resolution,
                "patch_grid": cue.patch_grid,
            }
            if cue.E_cue_patch is not None:
                save_dict["E_cue_patch"] = cue.E_cue_patch
            if cue.G_write_geo_patch is not None:
                save_dict["G_write_geo_patch"] = cue.G_write_geo_patch
            torch.save(save_dict, args.output)
            print(f"Saved cue output to {args.output}")
    finally:
        # -------------------------------------------------------------------
        # Cleanup
        # -------------------------------------------------------------------
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
