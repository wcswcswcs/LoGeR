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
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import torch

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
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1, help="-1 = all frames")
    p.add_argument("--stride", type=int, default=1)

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
    p.add_argument("--k_intra", type=int, default=3,
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
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # -- validate paths ------------------------------------------------------
    if args.config and not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # -----------------------------------------------------------------------
    # 1.  Collect images
    # -----------------------------------------------------------------------
    image_paths, temp_dir = collect_image_paths(
        args.input, args.start_frame, args.end_frame, args.stride,
    )
    if not image_paths:
        sys.exit("No images found.  Check --input path and frame range.")
    print(f"Collected {len(image_paths)} images.")

    # -----------------------------------------------------------------------
    # 2.  Load images
    # -----------------------------------------------------------------------
    target_w, target_h = args.resolution if args.resolution else (None, None)
    images = load_images(image_paths, target_w=target_w, target_h=target_h)
    print(f"Image tensor: {tuple(images.shape)}  (T, C, H, W)")

    # -----------------------------------------------------------------------
    # 3.  Build geometry backbone
    # -----------------------------------------------------------------------
    backbone_kwargs: dict = dict(
        device=args.device,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
    )
    if args.se3 is not None:
        backbone_kwargs["se3"] = args.se3

    print("Loading LoGeR model ...")
    t0 = time.time()
    backbone = LoGeRGeometryBackbone.from_config(
        checkpoint=args.checkpoint,
        config=args.config,
        **backbone_kwargs,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # -----------------------------------------------------------------------
    # 4.  Stage A -- geometry inference
    # -----------------------------------------------------------------------
    print("Running geometry inference (Stage A) ...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    geo: GeometryOutput = backbone.run(images)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_a = time.time() - t0
    fps_a = geo.num_frames / elapsed_a if elapsed_a > 0 else 0
    print(f"Stage A done in {elapsed_a:.2f}s  ({fps_a:.1f} FPS)")
    print_geometry_output(geo)

    # -----------------------------------------------------------------------
    # 5.  Stage B -- dynamic cue extraction
    # -----------------------------------------------------------------------
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

    print("Running dynamic cue extraction (Stage B) ...")
    t0 = time.time()

    cue: CueOutput = extractor.run(geo)

    elapsed_b = time.time() - t0
    print(f"Stage B done in {elapsed_b:.2f}s")
    print_cue_output(cue)

    # -----------------------------------------------------------------------
    # 6.  Optionally save .pt
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
