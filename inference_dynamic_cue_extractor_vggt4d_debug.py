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
        --chunk_size 32 \
        --sigma_pt 0.5 \
        --alpha_1 0.5 \
        --output_video results/taylor_cues_k_intra_4.mp4

    如果要开全层 debug: --debug_attention_vis
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

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
    p.add_argument("--chunk_overlap", type=int, default=2,
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
    p.add_argument("--debug_attention_vis", action="store_true",
                   help="Enable full query/key frame-attention debug visualisations for all 18 frame-attention layers.")
    p.add_argument("--reset_every", type=int, default=0)
    p.add_argument("--se3", action="store_true", default=None)

    # -- Dynamic Cue Extractor -----------------------------------------------
    p.add_argument("--k_intra", type=int, default=10,
                   help="Temporal window-width parameter; support candidates come from [t-k_intra//2, t+k_intra//2], and up to 4 views are sampled uniformly from that window.")
    p.add_argument("--disable_attention_prior", action="store_true",
                   help="Ignore Stage-A attention priors and fall back to uniform local support sampling.")
    p.add_argument("--support_time_decay", type=float, default=2.0,
                   help="Temporal decay used when scoring candidate support frames inside the local window.")
    p.add_argument("--support_temporal_weight", type=float, default=0.35,
                   help="Support ranking weight on temporal proximity.")
    p.add_argument("--support_affinity_weight", type=float, default=0.45,
                   help="Support ranking weight on frame-level attention affinity.")
    p.add_argument("--support_static_weight", type=float, default=0.20,
                   help="Support ranking weight on patch-level static overlap from attention priors.")
    p.add_argument("--sigma_pt", type=float, default=0.25,
                   help="Scale for point-residual -> consistency kernel.")
    p.add_argument("--tau_occ", type=float, default=0.05,
                   help="Depth-difference threshold for occlusion detection.")
    p.add_argument("--alpha_1", type=float, default=0.8,
                   help="C_dyn: weight on (1 - C_stat).")
    p.add_argument("--alpha_3", type=float, default=0.5,
                   help="C_dyn: subtract weight on C_occ.")
    p.add_argument("--attn_stat_fusion_weight", type=float, default=0.35,
                   help="Blend weight for attention-aware consistency into C_stat.")
    p.add_argument("--attn_dyn_weight", type=float, default=0.30,
                   help="Legacy compatibility option; the current implicit branch uses the Stage-A attention feature directly without extra scalar attenuation.")
    p.add_argument("--attn_gate_power", type=float, default=1.0,
                   help="Legacy option kept for compatibility; not used in the current raw-average implicit branch.")
    p.add_argument("--attn_debias_kernel", type=int, default=7,
                   help="Legacy option kept for compatibility; not used in the current raw-average implicit branch.")
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

    # -- VGGT4D alignment debug ----------------------------------------------
    p.add_argument("--run_vggt4d_alignment", action="store_true",
                   help="Run third_party/VGGT4D on the same clip, then auto-select LoGeR global-attn layers by matching VGGT4D family signals.")
    p.add_argument("--vggt4d_checkpoint", default="third_party/VGGT4D/ckpts/model_tracker_fixed_e20.pt",
                   help="Path to VGGT4D checkpoint used for alignment.")
    p.add_argument("--vggt4d_device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device for VGGT4D alignment run.")
    p.add_argument("--vggt4d_max_frames", type=int, default=64,
                   help="Max frames used for VGGT4D alignment (0 = use all merged frames).")
    p.add_argument("--vggt4d_chunk_size", type=int, default=24,
                   help="Chunk size used by VGGT4D alignment. Set <=0 to force a single forward on all selected frames.")
    p.add_argument("--vggt4d_chunk_overlap", type=int, default=0,
                   help="Chunk overlap used by VGGT4D alignment. 0 = auto, which promotes overlap to 2 * max(|offset|).")
    p.add_argument("--vggt4d_offsets", default="-6,-4,-2,2,4,6",
                   help="Comma-separated temporal offsets used by VGGT4D-style family extraction.")
    p.add_argument("--vggt4d_select_topk", type=int, default=4,
                   help="Top-k layers kept per family during automatic LoGeR layer selection.")

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
    if cue.C_dyn_explicit is not None:
        print(_stat_line("C_dyn_explicit", cue.C_dyn_explicit))
    if cue.C_dyn_implicit is not None:
        print(_stat_line("C_dyn_implicit", cue.C_dyn_implicit))
    if cue.C_dyn_fusion_max is not None:
        print(_stat_line("C_dyn_fusion_max", cue.C_dyn_fusion_max))
    if cue.C_dyn_fusion_soft_or is not None:
        print(_stat_line("C_dyn_fusion_soft_or", cue.C_dyn_fusion_soft_or))
    if cue.C_dyn_fusion_avg is not None:
        print(_stat_line("C_dyn_fusion_avg", cue.C_dyn_fusion_avg))
    if cue.C_dyn_fusion_addclip is not None:
        print(_stat_line("C_dyn_fusion_addclip", cue.C_dyn_fusion_addclip))

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
        sm = cue.debug.get("support_mass_per_frame")
        if sm is not None:
            print(f"    support mass        : min={sm.min().item():.4f}  max={sm.max().item():.4f}")
        ss = cue.debug.get("support_score_per_frame")
        if ss is not None:
            print(f"    support score sum   : min={ss.min().item():.4f}  max={ss.max().item():.4f}")
        ap = cue.debug.get("attention_prior_used")
        if ap is not None:
            print(f"    attention prior     : {'enabled' if ap else 'disabled'}")
        fa = cue.debug.get("frame_attention_mean")
        if fa is not None:
            print(f"    frame_attn mean     : {fa:.6f}")
        wc = cue.debug.get("weighted_consistency_mean")
        if wc is not None:
            print(f"    weighted consistency: {wc:.6f}")
        gc = cue.debug.get("geometry_consistency_mean")
        if gc is not None:
            print(f"    geometry consistency: {gc:.6f}")
        ed = cue.debug.get("explicit_dynamic_mean")
        if ed is not None:
            print(f"    explicit dynamic    : {ed:.6f}")
        adm = cue.debug.get("attention_dynamic_mean")
        if adm is not None:
            print(f"    attn feature mean   : {adm:.6f}")
        addm = cue.debug.get("attention_dynamic_used_mean")
        if addm is not None:
            print(f"    attn implicit mean  : {addm:.6f}")
        ag = cue.debug.get("attention_support_mean")
        if ag is not None:
            print(f"    attention support   : {ag:.6f}")

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
    c_dyn_explicit_parts = []
    c_dyn_implicit_parts = []
    c_dyn_fusion_max_parts = []
    c_dyn_fusion_soft_or_parts = []
    c_dyn_fusion_avg_parts = []
    c_dyn_fusion_addclip_parts = []
    debug = {"chunk_debug": []}

    use_patch = all(c.E_cue_patch is not None for c in chunk_cues)
    use_patch = use_patch and all(c.G_write_geo_patch is not None for c in chunk_cues)

    for i, c in enumerate(chunk_cues):
        drop = 0 if i == 0 else min(chunk_overlap, c.num_frames)
        if drop >= c.num_frames:
            continue
        e_cue_parts.append(c.E_cue[drop:])
        g_write_parts.append(c.G_write_geo[drop:])
        if c.C_dyn_explicit is not None:
            c_dyn_explicit_parts.append(c.C_dyn_explicit[drop:])
        if c.C_dyn_implicit is not None:
            c_dyn_implicit_parts.append(c.C_dyn_implicit[drop:])
        if c.C_dyn_fusion_max is not None:
            c_dyn_fusion_max_parts.append(c.C_dyn_fusion_max[drop:])
        if c.C_dyn_fusion_soft_or is not None:
            c_dyn_fusion_soft_or_parts.append(c.C_dyn_fusion_soft_or[drop:])
        if c.C_dyn_fusion_avg is not None:
            c_dyn_fusion_avg_parts.append(c.C_dyn_fusion_avg[drop:])
        if c.C_dyn_fusion_addclip is not None:
            c_dyn_fusion_addclip_parts.append(c.C_dyn_fusion_addclip[drop:])
        if use_patch:
            e_cue_patch_parts.append(c.E_cue_patch[drop:])
            g_write_patch_parts.append(c.G_write_geo_patch[drop:])
        debug["chunk_debug"].append(c.debug)

    if not e_cue_parts:
        e_cue_parts = [chunk_cues[0].E_cue]
        g_write_parts = [chunk_cues[0].G_write_geo]
        if chunk_cues[0].C_dyn_explicit is not None:
            c_dyn_explicit_parts = [chunk_cues[0].C_dyn_explicit]
        if chunk_cues[0].C_dyn_implicit is not None:
            c_dyn_implicit_parts = [chunk_cues[0].C_dyn_implicit]
        if chunk_cues[0].C_dyn_fusion_max is not None:
            c_dyn_fusion_max_parts = [chunk_cues[0].C_dyn_fusion_max]
        if chunk_cues[0].C_dyn_fusion_soft_or is not None:
            c_dyn_fusion_soft_or_parts = [chunk_cues[0].C_dyn_fusion_soft_or]
        if chunk_cues[0].C_dyn_fusion_avg is not None:
            c_dyn_fusion_avg_parts = [chunk_cues[0].C_dyn_fusion_avg]
        if chunk_cues[0].C_dyn_fusion_addclip is not None:
            c_dyn_fusion_addclip_parts = [chunk_cues[0].C_dyn_fusion_addclip]
        if use_patch:
            e_cue_patch_parts = [chunk_cues[0].E_cue_patch]
            g_write_patch_parts = [chunk_cues[0].G_write_geo_patch]

    E_cue = torch.cat(e_cue_parts, dim=0)
    G_write_geo = torch.cat(g_write_parts, dim=0)
    C_dyn_explicit = (
        torch.cat(c_dyn_explicit_parts, dim=0) if c_dyn_explicit_parts else None
    )
    C_dyn_implicit = (
        torch.cat(c_dyn_implicit_parts, dim=0) if c_dyn_implicit_parts else None
    )
    C_dyn_fusion_max = (
        torch.cat(c_dyn_fusion_max_parts, dim=0) if c_dyn_fusion_max_parts else None
    )
    C_dyn_fusion_soft_or = (
        torch.cat(c_dyn_fusion_soft_or_parts, dim=0) if c_dyn_fusion_soft_or_parts else None
    )
    C_dyn_fusion_avg = (
        torch.cat(c_dyn_fusion_avg_parts, dim=0) if c_dyn_fusion_avg_parts else None
    )
    C_dyn_fusion_addclip = (
        torch.cat(c_dyn_fusion_addclip_parts, dim=0) if c_dyn_fusion_addclip_parts else None
    )
    E_cue_patch = torch.cat(e_cue_patch_parts, dim=0) if use_patch else None
    G_write_geo_patch = torch.cat(g_write_patch_parts, dim=0) if use_patch else None

    merged = CueOutput(
        E_cue=E_cue,
        G_write_geo=G_write_geo,
        C_dyn_explicit=C_dyn_explicit,
        C_dyn_implicit=C_dyn_implicit,
        C_dyn_fusion_max=C_dyn_fusion_max,
        C_dyn_fusion_soft_or=C_dyn_fusion_soft_or,
        C_dyn_fusion_avg=C_dyn_fusion_avg,
        C_dyn_fusion_addclip=C_dyn_fusion_addclip,
        E_cue_patch=E_cue_patch,
        G_write_geo_patch=G_write_geo_patch,
        num_frames=E_cue.shape[0],
        spatial_resolution=chunk_cues[0].spatial_resolution,
        patch_grid=chunk_cues[0].patch_grid,
        debug=debug,
    )
    return merged


def merge_chunk_sequence_tensor(
    chunk_tensors: List[torch.Tensor | None],
    chunk_overlap: int,
) -> torch.Tensor | None:
    """Merge chunked [T, ...] tensors using the same drop policy as cues."""
    merged_parts: List[torch.Tensor] = []
    for i, tensor in enumerate(chunk_tensors):
        if tensor is None:
            continue
        num_frames = int(tensor.shape[0])
        if num_frames <= 0:
            continue
        drop = 0 if i == 0 else min(chunk_overlap, num_frames)
        if drop < num_frames:
            merged_parts.append(tensor[drop:])

    if not merged_parts:
        return None
    return torch.cat(merged_parts, dim=0)


def merge_chunk_attention_priors(
    frame_priors: List[torch.Tensor | None],
    dynamic_patches: List[torch.Tensor | None],
    chunk_overlap: int,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    """Merge Stage-A attention priors using the same chunk drop policy as cues."""
    merged_dynamic_parts: List[torch.Tensor] = []
    frame_blocks: List[Tuple[int, int, torch.Tensor]] = []
    total_frames = 0

    for i, (frame_prior, dyn_patch) in enumerate(zip(frame_priors, dynamic_patches)):
        num_frames = 0
        if dyn_patch is not None:
            num_frames = int(dyn_patch.shape[0])
        elif frame_prior is not None:
            num_frames = int(frame_prior.shape[0])
        if num_frames <= 0:
            continue

        drop = 0 if i == 0 else min(chunk_overlap, num_frames)
        keep = torch.arange(drop, num_frames)
        if keep.numel() == 0:
            continue

        kept_frames = int(keep.numel())
        if dyn_patch is not None:
            merged_dynamic_parts.append(dyn_patch[drop:])

        if frame_prior is not None:
            block = frame_prior.index_select(0, keep).index_select(1, keep)
            frame_blocks.append((total_frames, total_frames + kept_frames, block))

        total_frames += kept_frames

    merged_dynamic = (
        torch.cat(merged_dynamic_parts, dim=0) if merged_dynamic_parts else None
    )

    merged_frame = None
    if frame_blocks:
        dtype = frame_blocks[0][2].dtype
        merged_frame = torch.zeros(total_frames, total_frames, dtype=dtype)
        for start, end, block in frame_blocks:
            merged_frame[start:end, start:end] = block

    return merged_frame, merged_dynamic


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


def _prepare_layer_vis_map(
    map_2d: np.ndarray,
    mode: str = "fixed",
    temperature: float = 0.75,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    """Prepare a 2D map for visualization without changing the underlying stats.

    `fixed` keeps the original [0, 1] semantics. `contrast` applies a
    temperature-scaled z-score sigmoid so both unusually low and unusually high
    regions become easier to see. This is preferable to a pure spatial softmax
    for debug videos because some families (for example mean3) use low values
    as the dynamic cue.
    """
    map_2d = np.nan_to_num(map_2d, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    if mode == "fixed":
        return map_2d, (0.0, 1.0)
    if mode == "adaptive":
        return map_2d, None
    if mode == "contrast":
        mean = float(map_2d.mean())
        std = float(map_2d.std())
        z = (map_2d - mean) / max(std, 1e-6)
        z = np.clip(z / max(temperature, 1e-6), -6.0, 6.0)
        contrast = 1.0 / (1.0 + np.exp(-z))
        return contrast.astype(np.float32), (0.0, 1.0)
    raise ValueError(f"Unknown visualization mode: {mode}")


def _render_panel(title: str, img_bgr: np.ndarray) -> np.ndarray:
    panel = img_bgr.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        panel, title, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def _make_colorbar_legend(
    height: int,
    cmap: int = cv2.COLORMAP_TURBO,
    title: str = "Value",
    top_label: str = "1.00",
    mid_label: str = "0.50",
    bottom_label: str = "0.00",
    width: int = 96,
) -> np.ndarray:
    """Create a vertical colorbar legend panel."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (20, 20, 20), 1)
    cv2.putText(
        panel, title, (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )

    bar_top = 34
    bar_bottom = height - 18
    bar_h = max(bar_bottom - bar_top, 8)
    bar_x0 = 10
    bar_x1 = 30
    grad = np.linspace(1.0, 0.0, bar_h, dtype=np.float32)[:, None]
    grad = np.repeat(grad, max(bar_x1 - bar_x0, 1), axis=1)
    grad_color = _colorize_map(grad, cmap=cmap, value_range=(0.0, 1.0))
    panel[bar_top:bar_bottom, bar_x0:bar_x1] = grad_color[: bar_bottom - bar_top, : bar_x1 - bar_x0]
    cv2.rectangle(panel, (bar_x0, bar_top), (bar_x1, bar_bottom - 1), (255, 255, 255), 1)

    label_x = 40
    cv2.putText(
        panel, top_label, (label_x, bar_top + 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, mid_label, (label_x, bar_top + bar_h // 2 + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel, bottom_label, (label_x, bar_bottom - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return panel


def _append_colorbar_legend(
    frame: np.ndarray,
    cmap: int = cv2.COLORMAP_TURBO,
    title: str = "Value",
    top_label: str = "1.00",
    mid_label: str = "0.50",
    bottom_label: str = "0.00",
    gap: int = 4,
) -> np.ndarray:
    legend = _make_colorbar_legend(
        height=frame.shape[0],
        cmap=cmap,
        title=title,
        top_label=top_label,
        mid_label=mid_label,
        bottom_label=bottom_label,
    )
    return np.concatenate([frame, np.zeros((frame.shape[0], gap, 3), np.uint8), legend], axis=1)


def _summarize_patchvec_response(
    patchvec: torch.Tensor,
) -> torch.Tensor:
    """Convert a [T, H_tok, W_tok, D] raw patch-vector field into a scalar map.

    We project each patch vector onto the sequence-global centroid direction so
    the visualization reflects the exported raw q/k vectors themselves, rather
    than a new hand-crafted attention proxy.
    """
    if patchvec.dim() != 4:
        raise ValueError(f"Expected [T, H, W, D] patch vectors, got {tuple(patchvec.shape)}")

    T, H_tok, W_tok, dim = patchvec.shape
    flat = patchvec.reshape(T, H_tok * W_tok, dim).float()
    centroid = flat.mean(dim=(0, 1), keepdim=True)
    centroid = F.normalize(centroid, dim=-1)
    response = (flat * centroid).sum(dim=-1)
    lo = response.amin(dim=-1, keepdim=True)
    hi = response.amax(dim=-1, keepdim=True)
    response = (response - lo) / (hi - lo).clamp_min(1e-6)
    return response.reshape(T, H_tok, W_tok).clamp(0.0, 1.0)


def _summarize_patchvec_response_layers(
    patchvec_layers: torch.Tensor,
) -> torch.Tensor:
    """Convert [T, L, H_tok, W_tok, D] raw patch vectors into [T, L, H_tok, W_tok].

    Each layer uses its own sequence-global centroid direction so per-layer raw
    q/k responses can be compared directly in a layer-grid video.
    """
    if patchvec_layers.dim() != 5:
        raise ValueError(
            f"Expected [T, L, H, W, D] patch-vector layers, got {tuple(patchvec_layers.shape)}"
        )

    T, L, H_tok, W_tok, dim = patchvec_layers.shape
    flat = patchvec_layers.permute(1, 0, 2, 3, 4).reshape(L, T, H_tok * W_tok, dim).float()
    centroid = flat.mean(dim=(1, 2), keepdim=True)
    centroid = F.normalize(centroid, dim=-1)
    response = (flat * centroid).sum(dim=-1)
    lo = response.amin(dim=-1, keepdim=True)
    hi = response.amax(dim=-1, keepdim=True)
    response = (response - lo) / (hi - lo).clamp_min(1e-6)
    return response.reshape(L, T, H_tok, W_tok).permute(1, 0, 2, 3).contiguous().clamp(0.0, 1.0)


def _compute_dyn4d_token_gram_stats(
    global_q_raw_patchvec: torch.Tensor,
    global_k_raw_patchvec: torch.Tensor,
    window_radius: int,
) -> dict[str, torch.Tensor]:
    """Compute token-level Gram statistics from raw global q/k patch vectors.

    Inputs are expected to be merged sequence tensors with shape
    [T, H_tok, W_tok, D] or [T, L, H_tok, W_tok, D].
    """
    if global_q_raw_patchvec.dim() not in (4, 5) or global_k_raw_patchvec.dim() not in (4, 5):
        raise ValueError(
            "Expected raw global q/k patch vectors with shape "
            "[T, H_tok, W_tok, D] or [T, L, H_tok, W_tok, D]."
        )
    if global_q_raw_patchvec.shape != global_k_raw_patchvec.shape:
        raise ValueError(
            f"global_q/global_k shape mismatch: "
            f"{tuple(global_q_raw_patchvec.shape)} vs {tuple(global_k_raw_patchvec.shape)}"
        )

    if global_q_raw_patchvec.dim() == 4:
        global_q_raw_patchvec = global_q_raw_patchvec.unsqueeze(1)
        global_k_raw_patchvec = global_k_raw_patchvec.unsqueeze(1)

    T, L, H_tok, W_tok, dim = global_q_raw_patchvec.shape
    num_patches = H_tok * W_tok
    if T <= 1 or num_patches <= 0:
        zero = torch.zeros(T, H_tok, W_tok, dtype=torch.float32, device=global_q_raw_patchvec.device)
        return {
            "qq_mean_patch": zero,
            "qk_var_patch": zero,
            "kk_mean_patch": zero,
        }

    q = F.normalize(
        global_q_raw_patchvec.permute(1, 0, 2, 3, 4).reshape(L, T, num_patches, dim).float(),
        dim=-1,
    )
    k = F.normalize(
        global_k_raw_patchvec.permute(1, 0, 2, 3, 4).reshape(L, T, num_patches, dim).float(),
        dim=-1,
    )

    qq_sum = torch.zeros(L, T, num_patches, device=q.device, dtype=q.dtype)
    kk_sum = torch.zeros_like(qq_sum)
    qk_sum = torch.zeros_like(qq_sum)
    qk_sumsq = torch.zeros_like(qq_sum)
    counts = torch.zeros(1, T, 1, device=q.device, dtype=q.dtype)

    for t in range(T):
        start = max(0, t - int(window_radius))
        end = min(T, t + int(window_radius) + 1)
        q_t = q[:, t]
        k_t = k[:, t]
        for s in range(start, end):
            if s == t:
                continue
            q_s = q[:, s]
            k_s = k[:, s]

            qq_scores = torch.matmul(q_t, q_s.transpose(-1, -2))
            qk_scores = torch.matmul(q_t, k_s.transpose(-1, -2))
            kk_scores = torch.matmul(k_t, k_s.transpose(-1, -2))

            qq_sum[:, t] += qq_scores.sum(dim=-1)
            qk_sum[:, t] += qk_scores.sum(dim=-1)
            qk_sumsq[:, t] += qk_scores.square().sum(dim=-1)
            kk_sum[:, t] += kk_scores.sum(dim=-1)
            counts[:, t] += num_patches

    counts = counts.clamp_min(1.0)
    qq_mean = ((qq_sum / counts) + 1.0) * 0.5
    kk_mean = ((kk_sum / counts) + 1.0) * 0.5
    qk_mean = qk_sum / counts
    qk_var = (qk_sumsq / counts) - qk_mean.square()
    qk_var = qk_var.clamp_min(0.0)

    qq_mean = qq_mean.reshape(L, T, H_tok, W_tok).mean(dim=0).clamp(0.0, 1.0)
    kk_mean = kk_mean.reshape(L, T, H_tok, W_tok).mean(dim=0).clamp(0.0, 1.0)
    qk_var = qk_var.reshape(L, T, H_tok, W_tok).mean(dim=0)

    qk_var_flat = qk_var.reshape(T, -1)
    qk_var_min = qk_var_flat.amin(dim=-1, keepdim=True)
    qk_var_max = qk_var_flat.amax(dim=-1, keepdim=True)
    qk_var_norm = (
        (qk_var_flat - qk_var_min)
        / (qk_var_max - qk_var_min).clamp_min(1e-6)
    ).reshape_as(qk_var).clamp(0.0, 1.0)

    return {
        "qq_mean_patch": qq_mean,
        "qk_var_patch": qk_var_norm,
        "kk_mean_patch": kk_mean,
    }


def _compute_dyn4d_token_gram_stats_per_layer(
    global_q_raw_patchvec_layers: torch.Tensor,
    global_k_raw_patchvec_layers: torch.Tensor,
    window_radius: int,
) -> dict[str, torch.Tensor]:
    """Compute per-layer token-level Gram statistics for all global-attn layers.

    Inputs:
      global_q_raw_patchvec_layers/global_k_raw_patchvec_layers: [T, L, H, W, D]

    Returns:
      Dict of [T, L, H, W] maps:
        - qq_mean_layers
        - qq_var_layers
        - kk_mean_layers
        - qk_var_layers
    """
    if global_q_raw_patchvec_layers.dim() != 5 or global_k_raw_patchvec_layers.dim() != 5:
        raise ValueError(
            "Expected per-layer raw global q/k stacks with shape [T, L, H, W, D]."
        )
    if global_q_raw_patchvec_layers.shape != global_k_raw_patchvec_layers.shape:
        raise ValueError(
            f"global_q/global_k layer-stack shape mismatch: "
            f"{tuple(global_q_raw_patchvec_layers.shape)} vs {tuple(global_k_raw_patchvec_layers.shape)}"
        )

    T, L, H_tok, W_tok, D = global_q_raw_patchvec_layers.shape
    num_patches = H_tok * W_tok
    q = F.normalize(
        global_q_raw_patchvec_layers.permute(1, 0, 2, 3, 4).reshape(L, T, num_patches, D).float(),
        dim=-1,
    )
    k = F.normalize(
        global_k_raw_patchvec_layers.permute(1, 0, 2, 3, 4).reshape(L, T, num_patches, D).float(),
        dim=-1,
    )

    qq_sum = torch.zeros(L, T, num_patches, device=q.device, dtype=q.dtype)
    qq_sumsq = torch.zeros_like(qq_sum)
    kk_sum = torch.zeros_like(qq_sum)
    qk_sum = torch.zeros_like(qq_sum)
    qk_sumsq = torch.zeros_like(qq_sum)
    counts = torch.zeros(1, T, 1, device=q.device, dtype=q.dtype)

    for t in range(T):
        start = max(0, t - int(window_radius))
        end = min(T, t + int(window_radius) + 1)
        q_t = q[:, t]
        k_t = k[:, t]
        for s in range(start, end):
            if s == t:
                continue
            q_s = q[:, s]
            k_s = k[:, s]

            qq_scores = torch.matmul(q_t, q_s.transpose(-1, -2))
            kk_scores = torch.matmul(k_t, k_s.transpose(-1, -2))
            qk_scores = torch.matmul(q_t, k_s.transpose(-1, -2))

            qq_sum[:, t] += qq_scores.sum(dim=-1)
            qq_sumsq[:, t] += qq_scores.square().sum(dim=-1)
            kk_sum[:, t] += kk_scores.sum(dim=-1)
            qk_sum[:, t] += qk_scores.sum(dim=-1)
            qk_sumsq[:, t] += qk_scores.square().sum(dim=-1)
            counts[:, t] += num_patches

    counts = counts.clamp_min(1.0)
    qq_mean_raw = qq_sum / counts
    kk_mean_raw = kk_sum / counts
    qk_mean_raw = qk_sum / counts
    qq_var = (qq_sumsq / counts) - qq_mean_raw.square()
    qk_var = (qk_sumsq / counts) - qk_mean_raw.square()
    qq_var = qq_var.clamp_min(0.0)
    qk_var = qk_var.clamp_min(0.0)

    qq_mean = ((qq_mean_raw + 1.0) * 0.5).reshape(L, T, H_tok, W_tok).clamp(0.0, 1.0)
    kk_mean = ((kk_mean_raw + 1.0) * 0.5).reshape(L, T, H_tok, W_tok).clamp(0.0, 1.0)
    qq_var = qq_var.reshape(L, T, H_tok, W_tok)
    qk_var = qk_var.reshape(L, T, H_tok, W_tok)

    def _normalize_per_layer_frame(var_map: torch.Tensor) -> torch.Tensor:
        flat = var_map.reshape(L, T, -1)
        vmin = flat.amin(dim=-1, keepdim=True)
        vmax = flat.amax(dim=-1, keepdim=True)
        return (
            (flat - vmin)
            / (vmax - vmin).clamp_min(1e-6)
        ).reshape_as(var_map).clamp(0.0, 1.0)

    qq_var_norm = _normalize_per_layer_frame(qq_var)
    qk_var_norm = _normalize_per_layer_frame(qk_var)

    return {
        "qq_mean_layers": qq_mean.permute(1, 0, 2, 3).contiguous(),
        "qq_var_layers": qq_var_norm.permute(1, 0, 2, 3).contiguous(),
        "kk_mean_layers": kk_mean.permute(1, 0, 2, 3).contiguous(),
        "qk_var_layers": qk_var_norm.permute(1, 0, 2, 3).contiguous(),
    }


def _compose_dyn4d_from_components(
    qq_mean_patch: torch.Tensor,
    qk_var_patch: torch.Tensor,
    kk_mean_patch: torch.Tensor,
    weights: Tuple[float, float, float],
) -> torch.Tensor:
    """Compose final 4D_dyn from normalized Gram components."""
    w_qq, w_qk, w_kk = weights
    dyn4d_raw = (
        w_qq * (1.0 - qq_mean_patch)
        + w_qk * qk_var_patch
        + w_kk * (1.0 - kk_mean_patch)
    ).clamp(0.0, 1.0)
    dyn4d_flat = dyn4d_raw.reshape(dyn4d_raw.shape[0], -1)
    dyn4d_min = dyn4d_flat.amin(dim=-1, keepdim=True)
    dyn4d_max = dyn4d_flat.amax(dim=-1, keepdim=True)
    dyn4d = (
        (dyn4d_flat - dyn4d_min)
        / (dyn4d_max - dyn4d_min).clamp_min(1e-6)
    ).reshape_as(dyn4d_raw)
    return dyn4d.clamp(0.0, 1.0)


def _map_vggt4d_layer_range(
    total_layers: int,
    start: int,
    end: int,
    reference_total_layers: int = 24,
) -> torch.Tensor:
    """Map VGGT4D layer ranges onto LoGeR's available global-layer count."""
    if total_layers <= 0:
        return torch.empty(0, dtype=torch.long)
    mapped_start = int(np.floor(start / reference_total_layers * total_layers))
    mapped_end = int(np.ceil(end / reference_total_layers * total_layers))
    mapped_start = max(0, min(mapped_start, total_layers - 1))
    mapped_end = max(mapped_start + 1, min(mapped_end, total_layers))
    return torch.arange(mapped_start, mapped_end, dtype=torch.long)


def _normalize_map_2d_tensor(attn_map: torch.Tensor) -> torch.Tensor:
    attn_min = attn_map.min()
    attn_max = attn_map.max()
    return (attn_map - attn_min) / (attn_max - attn_min + 1e-6)


def _normalize_map_sequence(seq_map: torch.Tensor) -> torch.Tensor:
    """Per-frame min-max normalize a [T, H, W] tensor to [0, 1]."""
    if seq_map.dim() != 3:
        raise ValueError(f"Expected [T, H, W], got {tuple(seq_map.shape)}")
    flat = seq_map.reshape(seq_map.shape[0], -1)
    vmin = flat.amin(dim=-1, keepdim=True)
    vmax = flat.amax(dim=-1, keepdim=True)
    out = (flat - vmin) / (vmax - vmin).clamp_min(1e-6)
    return out.reshape_as(seq_map).clamp(0.0, 1.0)


def _build_patch_rgb_feature(
    images: torch.Tensor,
    patch_hw: tuple[int, int],
) -> torch.Tensor:
    """Downsample RGB images to patch-grid features [T, H_tok, W_tok, 3]."""
    patch_rgb = F.interpolate(
        images.float(),
        size=patch_hw,
        mode="bilinear",
        align_corners=False,
    )
    return patch_rgb.permute(0, 2, 3, 1).contiguous().clamp(0.0, 1.0)


def _pool_pointmap_to_patch_grid(
    pointmap: torch.Tensor,
    patch_hw: tuple[int, int],
) -> torch.Tensor:
    """Average-pool [T,H,W,C] pointmaps to [T,H_tok,W_tok,C]."""
    if pointmap.dim() != 4:
        raise ValueError(f"Expected [T,H,W,C], got {tuple(pointmap.shape)}")
    pooled = F.interpolate(
        pointmap.permute(0, 3, 1, 2).float(),
        size=patch_hw,
        mode="area",
    )
    return pooled.permute(0, 2, 3, 1).contiguous()


def _estimate_intrinsics_from_local_points(local_points: torch.Tensor) -> torch.Tensor:
    """Estimate per-frame intrinsics from camera-space pointmaps on the same grid."""
    if local_points.dim() != 4:
        raise ValueError(f"Expected [T,H,W,3], got {tuple(local_points.shape)}")
    T, H, W, _ = local_points.shape
    device = local_points.device
    dtype = local_points.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype) + 0.5,
        torch.arange(W, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    Ks = []
    for t in range(T):
        pts = local_points[t]
        z = pts[..., 2]
        x_over_z = pts[..., 0] / z.clamp_min(1e-6)
        y_over_z = pts[..., 1] / z.clamp_min(1e-6)
        valid_x = torch.isfinite(x_over_z) & torch.isfinite(xx) & (z > 1e-6)
        valid_y = torch.isfinite(y_over_z) & torch.isfinite(yy) & (z > 1e-6)

        def _solve(coord: torch.Tensor, ratio: torch.Tensor, valid: torch.Tensor, fallback_scale: float):
            if valid.sum() < 8:
                return torch.tensor(fallback_scale, device=device, dtype=dtype), torch.tensor(
                    (W if coord is xx else H) * 0.5,
                    device=device,
                    dtype=dtype,
                )
            a = torch.stack([ratio[valid], torch.ones_like(ratio[valid])], dim=-1)
            b = coord[valid]
            sol = torch.linalg.lstsq(a, b.unsqueeze(-1)).solution.squeeze(-1)
            return sol[0], sol[1]

        fx, cx = _solve(xx, x_over_z, valid_x, float(W))
        fy, cy = _solve(yy, y_over_z, valid_y, float(H))
        K = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        Ks.append(K)
    return torch.stack(Ks, dim=0)


def _project_world_points(
    pts_world: torch.Tensor,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> torch.Tensor:
    """Project world points [N,3] into all cameras -> [T,N,3] (u,v,z_cam)."""
    world2cam = torch.inverse(cam2world)[:, None, :3, :]
    pts_h = torch.cat([pts_world, torch.ones_like(pts_world[:, :1])], dim=-1)
    pts_cam = torch.einsum("tnij,nj->tni", world2cam, pts_h)
    proj = torch.einsum("tij,tnj->tni", intrinsics, pts_cam)
    uv = proj[..., :2] / proj[..., 2:3].clamp_min(1e-6)
    return torch.cat([uv, pts_cam[..., 2:3]], dim=-1)


def _cluster_attention_maps_cv2(
    feature: torch.Tensor,
    dynamic_map: torch.Tensor,
    n_clusters: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LoGeR-friendly replacement for the demo's cluster_attention_maps.

    We use OpenCV KMeans because sklearn is not available in this environment.
    """
    if feature.dim() != 4 or dynamic_map.dim() != 3:
        raise ValueError(
            f"Expected feature [T,H,W,C] and dynamic_map [T,H,W], got "
            f"{tuple(feature.shape)} and {tuple(dynamic_map.shape)}"
        )
    T, H, W, C = feature.shape
    feature_np = feature.detach().cpu().numpy().astype(np.float32)
    dynamic_np = dynamic_map.detach().cpu().numpy().astype(np.float32)
    flat_feature = feature_np.reshape(-1, C)
    flat_dynamic = dynamic_np.reshape(-1)
    num_samples = flat_feature.shape[0]
    if num_samples <= 1:
        labels = torch.zeros((T, H, W), dtype=torch.long)
        return dynamic_map.clone(), labels

    k = int(max(2, min(n_clusters, num_samples)))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.1,
    )
    _, labels_np, _ = cv2.kmeans(
        flat_feature,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels_np = labels_np.reshape(-1)
    cluster_scores = np.zeros(k, dtype=np.float32)
    for cluster_id in range(k):
        mask = labels_np == cluster_id
        if np.any(mask):
            cluster_scores[cluster_id] = float(flat_dynamic[mask].mean())
    clustered_np = cluster_scores[labels_np].reshape(T, H, W)
    clustered = torch.from_numpy(clustered_np).float()
    clustered = _normalize_map_sequence(clustered)
    cluster_labels = torch.from_numpy(labels_np.reshape(T, H, W)).long()
    return clustered, cluster_labels


def _adaptive_multiotsu_variance_np(
    img: np.ndarray,
    max_classes: int = 4,
    num_bins: int = 64,
) -> float:
    """Dependency-free approximation of adaptive_multiotsu_variance.

    Returns the highest threshold of the best 2-4 class partition, matching the
    demo's behavior of using the top class as the dynamic mask.
    """
    values = np.asarray(img, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.5
    values = np.clip(values, 0.0, 1.0)
    hist, bin_edges = np.histogram(values, bins=num_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    prob = hist / total
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    def _segment_stats(start: int, end: int) -> tuple[float, float]:
        w = omega[end] - (omega[start - 1] if start > 0 else 0.0)
        if w <= 1e-12:
            return 0.0, 0.0
        m = (mu[end] - (mu[start - 1] if start > 0 else 0.0)) / w
        return w, m

    best_score = -float("inf")
    best_threshold = 0.5

    # 2 classes
    for t0 in range(0, num_bins - 1):
        parts = [(0, t0), (t0 + 1, num_bins - 1)]
        score = 0.0
        valid = True
        for s, e in parts:
            w, m = _segment_stats(s, e)
            if w <= 1e-12:
                valid = False
                break
            score += w * (m - mu_t) ** 2
        score /= np.sqrt(2.0)
        if valid and score > best_score:
            best_score = score
            best_threshold = float(bin_edges[t0 + 1])

    # 3 classes
    for t0 in range(0, num_bins - 2):
        for t1 in range(t0 + 1, num_bins - 1):
            parts = [(0, t0), (t0 + 1, t1), (t1 + 1, num_bins - 1)]
            score = 0.0
            valid = True
            for s, e in parts:
                w, m = _segment_stats(s, e)
                if w <= 1e-12:
                    valid = False
                    break
                score += w * (m - mu_t) ** 2
            score /= np.sqrt(3.0)
            if valid and score > best_score:
                best_score = score
                best_threshold = float(bin_edges[t1 + 1])

    # 4 classes
    for t0 in range(0, num_bins - 3):
        for t1 in range(t0 + 1, num_bins - 2):
            for t2 in range(t1 + 1, num_bins - 1):
                parts = [(0, t0), (t0 + 1, t1), (t1 + 1, t2), (t2 + 1, num_bins - 1)]
                score = 0.0
                valid = True
                for s, e in parts:
                    w, m = _segment_stats(s, e)
                    if w <= 1e-12:
                        valid = False
                        break
                    score += w * (m - mu_t) ** 2
                score /= np.sqrt(4.0)
                if valid and score > best_score:
                    best_score = score
                    best_threshold = float(bin_edges[t2 + 1])

    return float(np.clip(best_threshold, 0.0, 1.0))


class LoGeRRefineDynMask:
    """LoGeR-adapted patch-grid refinement modeled after 4DVGGT RefineDynMask."""

    def __init__(
        self,
        images: torch.Tensor,
        world_points: torch.Tensor,
        local_points: torch.Tensor,
        coarse_mask: torch.Tensor,
        cam2world: torch.Tensor,
        coarse_map: torch.Tensor,
        frame_gate: torch.Tensor | None = None,
        cue_patch: torch.Tensor | None = None,
    ):
        self.images = images.float()
        self.world_points = world_points.float()
        self.local_points = local_points.float()
        self.coarse_map = coarse_map.float()
        self.coarse_mask = coarse_mask.float()
        self.cam2world = cam2world.float()
        self.frame_gate = frame_gate.float() if frame_gate is not None else None
        self.cue_patch = cue_patch.float() if cue_patch is not None else None
        self.intrinsics = _estimate_intrinsics_from_local_points(self.local_points)

    def _grid_sample_depth(self, depths: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        h, w = depths.shape[-2:]
        grid = uv.clone()
        grid[..., 0] = grid[..., 0] / max(w - 1, 1)
        grid[..., 1] = grid[..., 1] / max(h - 1, 1)
        grid[..., 0] = grid[..., 0] * 2 - 1
        grid[..., 1] = grid[..., 1] * 2 - 1
        return F.grid_sample(depths, grid, mode="nearest", align_corners=True)

    def _grid_sample_mask(self, masks: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        h, w = masks.shape[-2:]
        grid = uv.clone()
        grid[..., 0] = grid[..., 0] / max(w - 1, 1)
        grid[..., 1] = grid[..., 1] / max(h - 1, 1)
        grid[..., 0] = grid[..., 0] * 2 - 1
        grid[..., 1] = grid[..., 1] * 2 - 1
        out = F.grid_sample(masks.float(), grid, mode="bilinear", align_corners=True)
        return out > 0.5

    def _grid_sample_rgb(self, rgbs: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        h, w = rgbs.shape[-2:]
        grid = uv.clone()
        grid[..., 0] = grid[..., 0] / max(w - 1, 1)
        grid[..., 1] = grid[..., 1] / max(h - 1, 1)
        grid[..., 0] = grid[..., 0] * 2 - 1
        grid[..., 1] = grid[..., 1] * 2 - 1
        return F.grid_sample(rgbs.float(), grid, mode="bilinear", align_corners=True)

    def _compute_dyn_loss(
        self,
        cam_id: int,
        pts_world: torch.Tensor,
        rgb: torch.Tensor,
        labels: torch.Tensor,
        dyn_labels: torch.Tensor,
    ) -> list[tuple[int, float, float, float]]:
        n_img, h_img, w_img, _ = self.images.shape
        label_losses = []
        other_cam_id = torch.tensor([i for i in range(n_img) if i != cam_id], dtype=torch.long, device=self.images.device)
        if other_cam_id.numel() == 0:
            return []
        other_depths = self.local_points[other_cam_id][..., 2][:, None, ...]
        other_dyn_masks = self.coarse_mask[other_cam_id][:, None, ...]
        other_rgbs = self.images[other_cam_id].permute(0, 3, 1, 2)
        proj_all = _project_world_points(pts_world, self.intrinsics[other_cam_id], self.cam2world[other_cam_id])

        for label in dyn_labels.tolist():
            pick_mask = labels == label
            if pick_mask.sum() == 0:
                continue
            pick_proj = proj_all[:, pick_mask]
            pick_rgb = rgb[pick_mask]
            valid_width = (pick_proj[..., 0] > 0) & (pick_proj[..., 0] < w_img)
            valid_height = (pick_proj[..., 1] > 0) & (pick_proj[..., 1] < h_img)
            valid_depth = pick_proj[..., 2] > 0
            valid_proj = valid_width & valid_height & valid_depth

            sample_uv = pick_proj[:, None, :, :2]
            sample_depths = self._grid_sample_depth(other_depths, sample_uv)
            sample_dyn_masks = self._grid_sample_mask(other_dyn_masks, sample_uv)
            sample_rgbs = self._grid_sample_rgb(other_rgbs, sample_uv)

            sample_depths = rearrange(sample_depths, "n_cam 1 1 n_pick -> n_cam n_pick")
            sample_dyn_masks = rearrange(sample_dyn_masks, "n_cam 1 1 n_pick -> n_cam n_pick")
            sample_rgbs = rearrange(sample_rgbs, "n_cam c 1 n_pick -> n_cam n_pick c")

            visible_mask = pick_proj[..., 2] - 0.01 < sample_depths
            loss_mask = visible_mask & (~sample_dyn_masks) & valid_proj

            num_loss_points = loss_mask.sum()
            total_sample_points = max((n_img - 1) * int(pick_mask.sum().item()), 1)
            if (num_loss_points.float() / float(total_sample_points)) < 0.05:
                label_losses.append((label, 1e10, 1e10, 1e10))
                continue

            depth_diff = torch.abs(pick_proj[..., 2] - sample_depths)
            rgb_diff = torch.abs(pick_rgb.unsqueeze(0) - sample_rgbs)
            valid_depth_diff = depth_diff[loss_mask]
            valid_rgb_diff = rgb_diff[loss_mask]
            depth_loss = valid_depth_diff.sum() / loss_mask.sum().clamp_min(1)
            rgb_loss = valid_rgb_diff.sum() / loss_mask.sum().clamp_min(1)
            total_loss = depth_loss + rgb_loss / 3.0
            label_losses.append((label, float(depth_loss), float(rgb_loss), float(total_loss)))
        return label_losses

    def refine_masks(self) -> torch.Tensor:
        T, H, W = self.coarse_map.shape
        refined_masks = []
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_dilate = np.ones((3, 3), np.uint8)

        rgb_patch = self.images
        pts_patch = self.world_points
        for cam_id in range(T):
            pts = rearrange(pts_patch[cam_id], "h w c -> (h w) c")
            rgb = rearrange(rgb_patch[cam_id], "h w c -> (h w) c")
            coarse_mask = rearrange(self.coarse_mask[cam_id], "h w -> (h w)") > 0.5

            valid_pts = torch.isfinite(pts).all(dim=-1)
            dyn_idx = valid_pts & coarse_mask
            if dyn_idx.sum() <= 2:
                refined_masks.append(_normalize_map_2d_tensor(self.coarse_map[cam_id]))
                continue

            dyn_pts = pts[dyn_idx].detach().cpu().numpy().astype(np.float32)
            k = int(max(2, min(30, dyn_pts.shape[0])))
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                50,
                0.1,
            )
            _, dyn_labels_np, _ = cv2.kmeans(
                dyn_pts,
                k,
                None,
                criteria,
                3,
                cv2.KMEANS_PP_CENTERS,
            )
            dyn_labels = torch.from_numpy(dyn_labels_np.reshape(-1)).long().to(pts.device)
            pts_labels = torch.full((pts.shape[0],), -2, dtype=torch.long, device=pts.device)
            pts_labels[valid_pts & (~coarse_mask)] = -1
            pts_labels[dyn_idx] = dyn_labels
            unique_dyn_labels = torch.unique(dyn_labels)

            label_losses = self._compute_dyn_loss(cam_id, pts, rgb, pts_labels, unique_dyn_labels)
            selected_labels = torch.tensor(
                [label for label, _, _, loss in label_losses if loss > 0.1],
                dtype=torch.long,
                device=pts.device,
            )
            if selected_labels.numel() == 0:
                refine_dyn_mask = coarse_mask.clone()
            else:
                refine_dyn_mask = torch.isin(pts_labels, selected_labels)
            refine_dyn_mask = rearrange(refine_dyn_mask, "(h w) -> h w", h=H, w=W)

            mask_np = (refine_dyn_mask.detach().cpu().numpy().astype(np.uint8) * 255)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            mask_np = cv2.dilate(mask_np, kernel_dilate, iterations=1)
            mask_t = torch.from_numpy((mask_np > 0).astype(np.float32)).to(self.coarse_map.device)

            score = self.coarse_map[cam_id] * (0.60 + 0.40 * mask_t)
            if self.frame_gate is not None:
                score = score * (0.70 + 0.30 * self.frame_gate[cam_id])
            if self.cue_patch is not None:
                dyn_patch = self.cue_patch[cam_id, :, :, 1]
                occ_patch = self.cue_patch[cam_id, :, :, 2]
                unc_patch = self.cue_patch[cam_id, :, :, 3]
                score = torch.maximum(
                    score,
                    dyn_patch * (1.0 - occ_patch) * (1.0 - 0.5 * unc_patch) * (0.5 + 0.5 * mask_t),
                )
                score = score * (1.0 - 0.25 * occ_patch)
            score = F.avg_pool2d(score[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            refined_masks.append(_normalize_map_2d_tensor(score))

        return torch.stack(refined_masks, dim=0).cpu()


@torch.no_grad()
def _postprocess_dyn_map_like_vggt4d_demo(
    dyn_map: torch.Tensor,
    images: torch.Tensor | None = None,
    world_points: torch.Tensor | None = None,
    local_points: torch.Tensor | None = None,
    camera_poses: torch.Tensor | None = None,
    cue_patch: torch.Tensor | None = None,
) -> dict[str, object]:
    """Apply demo-style cluster/threshold/refine post-processing on a patch dyn map."""
    dyn_coarse = _normalize_map_sequence(dyn_map.float().cpu())
    patch_hw = tuple(dyn_coarse.shape[-2:])

    patch_rgb = _build_patch_rgb_feature(images, patch_hw).cpu() if images is not None else None
    patch_world = _pool_pointmap_to_patch_grid(world_points, patch_hw).cpu() if world_points is not None else None
    patch_local = _pool_pointmap_to_patch_grid(local_points, patch_hw).cpu() if local_points is not None else None
    patch_cam2world = camera_poses.float().cpu() if camera_poses is not None else None

    clustered_dyn = dyn_coarse
    cluster_labels = None
    if patch_rgb is not None:
        clustered_dyn, cluster_labels = _cluster_attention_maps_cv2(
            patch_rgb,
            dyn_coarse,
            n_clusters=64,
        )
        cluster_threshold = _adaptive_multiotsu_variance_np(clustered_dyn.numpy())
    else:
        cluster_threshold = 0.5
    binary_mask = (clustered_dyn > float(cluster_threshold)).float()

    if (
        patch_rgb is not None
        and patch_world is not None
        and patch_local is not None
        and patch_cam2world is not None
    ):
        refined_dyn = LoGeRRefineDynMask(
            images=patch_rgb,
            world_points=patch_world,
            local_points=patch_local,
            coarse_mask=binary_mask,
            cam2world=patch_cam2world,
            coarse_map=clustered_dyn,
            frame_gate=None,
            cue_patch=cue_patch.cpu() if cue_patch is not None else None,
        ).refine_masks()
    else:
        refined_dyn = clustered_dyn.cpu()

    return {
        "coarse_dyn_patch": dyn_coarse.cpu(),
        "cluster_dyn_patch": clustered_dyn.cpu(),
        "binary_mask_patch": binary_mask.cpu(),
        "cluster_threshold": float(cluster_threshold),
        "cluster_labels_patch": cluster_labels.cpu() if cluster_labels is not None else None,
        "refined_dyn_patch": refined_dyn.cpu(),
    }


def _vggt4d_extract_map(
    ref_id: int,
    global_q: torch.Tensor,
    global_k: torch.Tensor,
    patch_h: int,
    patch_w: int,
    layer_ids: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Direct adaptation of the helper maps from docs/4dvggt相关代码.py.

    Expected tensor shape after minimal adaptation:
    global_q/global_k: [T, L, H_head, N_tok, D].
    """
    device = global_q.device
    window = torch.tensor([-6, -4, -2, 2, 4, 6], device=device, dtype=torch.long)
    n_img = global_q.shape[0]
    src_ids = ref_id + window
    src_ids = src_ids[(src_ids >= 0) & (src_ids < n_img)]
    if src_ids.numel() == 0 or layer_ids.numel() == 0:
        return torch.zeros(patch_h, patch_w, device=device, dtype=global_q.dtype)

    q_ref = global_q[ref_id].unsqueeze(0)[:, layer_ids]
    k_ref = global_k[ref_id].unsqueeze(0)[:, layer_ids]
    q_src = global_q[src_ids][:, layer_ids]
    k_src = global_k[src_ids][:, layer_ids]

    if mode == "mean1":
        attn_map = q_ref @ q_src.transpose(-2, -1)
        attn_map = rearrange(
            attn_map,
            "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok",
            n_h=patch_h,
            n_w=patch_w,
        )
        attn_map = attn_map.mean(dim=(2, 3, 4))
    elif mode == "var1":
        attn_map = q_ref @ q_src.transpose(-2, -1)
        attn_map = rearrange(
            attn_map,
            "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok",
            n_h=patch_h,
            n_w=patch_w,
        )
        attn_map = attn_map.mean(dim=(2, 3)).std(dim=-1)
    elif mode == "mean2":
        attn_map = q_ref @ q_src.transpose(-2, -1)
        attn_map = rearrange(
            attn_map,
            "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok",
            n_h=patch_h,
            n_w=patch_w,
        )
        attn_map = attn_map.mean(dim=(2, 3, 4))
    elif mode == "mean3":
        attn_map = k_ref @ k_src.transpose(-2, -1)
        attn_map = rearrange(
            attn_map,
            "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok",
            n_h=patch_h,
            n_w=patch_w,
        )
        attn_map = attn_map.mean(dim=(2, 3, 4))
    elif mode == "var3":
        attn_map = q_ref @ k_src.transpose(-2, -1)
        attn_map = rearrange(
            attn_map,
            "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok",
            n_h=patch_h,
            n_w=patch_w,
        )
        attn_map = attn_map.mean(dim=(2, 3)).std(dim=-1)
    else:
        raise ValueError(f"Unsupported VGGT4D map mode: {mode}")

    return _normalize_map_2d_tensor(attn_map)


@torch.no_grad()
def _build_vggt4d_direct_variant(
    global_q_raw_patchvec_layers: torch.Tensor,
    global_k_raw_patchvec_layers: torch.Tensor,
) -> dict[str, object]:
    """Build a direct VGGT4D-style variant from exported raw global q/k layers.

    We follow the helper functions in docs/4dvggt相关代码.py as closely as
    possible, with two unavoidable adaptations:
    1. LoGeR has a different number of global-attention layers, so we map the
       original layer ranges by depth ratio.
    2. Our exported q/k are currently head-averaged, so we insert a dummy head
       dimension of size 1 and reuse the same formulas.
    """
    if global_q_raw_patchvec_layers.dim() != 5 or global_k_raw_patchvec_layers.dim() != 5:
        raise ValueError(
            "Expected per-layer raw global q/k stacks with shape [T, L, H, W, D]."
        )
    if global_q_raw_patchvec_layers.shape != global_k_raw_patchvec_layers.shape:
        raise ValueError(
            f"global_q/global_k layer-stack shape mismatch: "
            f"{tuple(global_q_raw_patchvec_layers.shape)} vs {tuple(global_k_raw_patchvec_layers.shape)}"
        )

    T, L, H_tok, W_tok, D = global_q_raw_patchvec_layers.shape
    device = torch.device("cuda" if torch.cuda.is_available() else global_q_raw_patchvec_layers.device)
    q = global_q_raw_patchvec_layers.to(device=device, dtype=torch.float32)
    k = global_k_raw_patchvec_layers.to(device=device, dtype=torch.float32)

    q = F.normalize(q.reshape(T, L, H_tok * W_tok, D), dim=-1).unsqueeze(2)
    k = F.normalize(k.reshape(T, L, H_tok * W_tok, D), dim=-1).unsqueeze(2)

    mean1_layers = _map_vggt4d_layer_range(L, 3, 8)
    var1_layers = _map_vggt4d_layer_range(L, 18, 20)
    mean2_layers = _map_vggt4d_layer_range(L, 17, 22)
    mean3_layers = _map_vggt4d_layer_range(L, 0, 1)
    var3_layers = _map_vggt4d_layer_range(L, 0, 1)

    dyn_maps = []
    mean1_maps = []
    var1_maps = []
    mean2_maps = []
    mean3_maps = []
    var3_maps = []
    for ref_id in range(T):
        mean1_map = _vggt4d_extract_map(ref_id, q, k, H_tok, W_tok, mean1_layers.to(device), "mean1")
        var1_map = _vggt4d_extract_map(ref_id, q, k, H_tok, W_tok, var1_layers.to(device), "var1")
        mean2_map = _vggt4d_extract_map(ref_id, q, k, H_tok, W_tok, mean2_layers.to(device), "mean2")
        mean3_map = _vggt4d_extract_map(ref_id, q, k, H_tok, W_tok, mean3_layers.to(device), "mean3")
        var3_map = _vggt4d_extract_map(ref_id, q, k, H_tok, W_tok, var3_layers.to(device), "var3")

        dyn_map = (1.0 - mean1_map) * (1.0 - var1_map) * mean2_map * (1.0 - mean3_map) * var3_map
        mean1_maps.append(mean1_map.detach().cpu())
        var1_maps.append(var1_map.detach().cpu())
        mean2_maps.append(mean2_map.detach().cpu())
        mean3_maps.append(mean3_map.detach().cpu())
        var3_maps.append(var3_map.detach().cpu())
        dyn_maps.append(_normalize_map_2d_tensor(dyn_map))

    dyn_maps = torch.stack(dyn_maps, dim=0).detach().cpu()
    return {
        "name": "vggt4d_direct",
        "title": "VGGT4D-direct",
        "radius": "window[-6,-4,-2,2,4,6]",
        "weights": "direct_formula",
        "layer_map": {
            "mean1": mean1_layers.tolist(),
            "var1": var1_layers.tolist(),
            "mean2": mean2_layers.tolist(),
            "mean3": mean3_layers.tolist(),
            "var3": var3_layers.tolist(),
        },
        "mean1_patch": torch.stack(mean1_maps, dim=0),
        "var1_patch": torch.stack(var1_maps, dim=0),
        "mean2_patch": torch.stack(mean2_maps, dim=0),
        "mean3_patch": torch.stack(mean3_maps, dim=0),
        "var3_patch": torch.stack(var3_maps, dim=0),
        "dyn4d_patch": dyn_maps,
    }


@torch.no_grad()
def _build_loger_lite_variant(
    global_q_raw_patchvec_layers: torch.Tensor,
    global_k_raw_patchvec_layers: torch.Tensor,
    global_layer_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    """Build a LoGeR-adapted lightweight 4D_dyn using only mean3 + var3 families.

    Current hypothesis:
    - low KK-mean is the most stable `mean3`-like cue
    - high QK-var is the most stable `var3`-like cue
    """
    stats = _compute_dyn4d_token_gram_stats_per_layer(
        global_q_raw_patchvec_layers,
        global_k_raw_patchvec_layers,
        window_radius=2,
    )

    if global_layer_ids is None:
        global_layer_ids = torch.arange(
            1, 2 * global_q_raw_patchvec_layers.shape[1] + 1, 2, dtype=torch.long,
        )
    else:
        global_layer_ids = global_layer_ids.detach().cpu().long()

    mean3_candidate_layers = [3, 5, 7, 15, 21]
    var3_candidate_layers = [1, 3, 11, 21, 23]

    mean3_idx = [
        int((global_layer_ids == layer_id).nonzero(as_tuple=False)[0, 0].item())
        for layer_id in mean3_candidate_layers
        if (global_layer_ids == layer_id).any()
    ]
    var3_idx = [
        int((global_layer_ids == layer_id).nonzero(as_tuple=False)[0, 0].item())
        for layer_id in var3_candidate_layers
        if (global_layer_ids == layer_id).any()
    ]

    if not mean3_idx:
        mean3_idx = list(range(min(5, int(global_layer_ids.numel()))))
    if not var3_idx:
        var3_idx = list(range(min(5, int(global_layer_ids.numel()))))

    kk_mean_layers = stats["kk_mean_layers"]
    qk_var_layers = stats["qk_var_layers"]
    mean3_hat = kk_mean_layers[:, mean3_idx].mean(dim=1)
    var3_hat = qk_var_layers[:, var3_idx].mean(dim=1)
    dyn_maps = []
    for t in range(mean3_hat.shape[0]):
        dyn_maps.append(_normalize_map_2d_tensor((1.0 - mean3_hat[t]) * var3_hat[t]))
    dyn_maps = torch.stack(dyn_maps, dim=0)

    return {
        "name": "loger_lite_m3v3",
        "title": "LoGeR-lite m3+v3",
        "radius": 2,
        "weights": "dyn=(1-mean3_hat)*var3_hat",
        "layer_map": {
            "mean3": global_layer_ids[mean3_idx].tolist(),
            "var3": global_layer_ids[var3_idx].tolist(),
        },
        "mean3_patch": mean3_hat.cpu(),
        "var3_patch": var3_hat.cpu(),
        "dyn4d_patch": dyn_maps.cpu(),
    }


@torch.no_grad()
def _build_vggt4d_custom_variant(
    global_q_raw_patchvec_layers: torch.Tensor,
    global_k_raw_patchvec_layers: torch.Tensor,
    global_layer_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    """Build a user-specified VGGT4D-style variant from selected LoGeR layers.

    Assumption:
      dyn = shallow_term * middle_term * deep_term,
    where
      shallow_term = (1 - KK_mean[L1]) * QK_var[L1]
      middle_term = mean_{L in [5,7,9,21]} (1 - QQ_mean[L])
      deep_term = QQ_var[L35]
    """
    stats = _compute_dyn4d_token_gram_stats_per_layer(
        global_q_raw_patchvec_layers,
        global_k_raw_patchvec_layers,
        window_radius=2,
    )

    if global_layer_ids is None:
        global_layer_ids = torch.arange(
            1, 2 * global_q_raw_patchvec_layers.shape[1] + 1, 2, dtype=torch.long,
        )
    else:
        global_layer_ids = global_layer_ids.detach().cpu().long()

    def _lookup_indices(target_layers: list[int]) -> list[int]:
        return [
            int((global_layer_ids == layer_id).nonzero(as_tuple=False)[0, 0].item())
            for layer_id in target_layers
            if (global_layer_ids == layer_id).any()
        ]

    shallow_idx = _lookup_indices([1])
    middle_idx = _lookup_indices([5, 7, 9, 21])
    deep_idx = _lookup_indices([35])

    if not shallow_idx:
        shallow_idx = [0]
    if not middle_idx:
        middle_idx = list(range(min(4, int(global_layer_ids.numel()))))
    if not deep_idx:
        deep_idx = [int(global_layer_ids.numel()) - 1]

    qq_mean_layers = stats["qq_mean_layers"]
    qq_var_layers = stats["qq_var_layers"]
    kk_mean_layers = stats["kk_mean_layers"]
    qk_var_layers = stats["qk_var_layers"]

    shallow_kk_mean = kk_mean_layers[:, shallow_idx].mean(dim=1)
    shallow_qk_var = qk_var_layers[:, shallow_idx].mean(dim=1)
    shallow_term = ((1.0 - shallow_kk_mean) * shallow_qk_var).clamp(0.0, 1.0)

    middle_qq_mean = qq_mean_layers[:, middle_idx].mean(dim=1)
    middle_term = (1.0 - middle_qq_mean).clamp(0.0, 1.0)

    deep_qq_var = qq_var_layers[:, deep_idx].mean(dim=1)
    deep_term = deep_qq_var.clamp(0.0, 1.0)

    dyn_maps = _normalize_map_sequence(
        (shallow_term * middle_term * deep_term).clamp(0.0, 1.0)
    ).cpu()

    return {
        "name": "vggt4d_custom",
        "title": "VGGT4D-custom",
        "radius": 2,
        "weights": (
            "dyn=((1-KKmean[L1])*QKvar[L1]) * "
            "mean(1-QQmean[L5,7,9,21]) * QQvar[L35]"
        ),
        "layer_map": {
            "shallow": global_layer_ids[shallow_idx].tolist(),
            "middle": global_layer_ids[middle_idx].tolist(),
            "deep": global_layer_ids[deep_idx].tolist(),
        },
        "shallow_kk_mean_patch": shallow_kk_mean.cpu(),
        "shallow_qk_var_patch": shallow_qk_var.cpu(),
        "shallow_term_patch": shallow_term.cpu(),
        "middle_qq_mean_patch": middle_qq_mean.cpu(),
        "middle_term_patch": middle_term.cpu(),
        "deep_qq_var_patch": deep_qq_var.cpu(),
        "deep_term_patch": deep_term.cpu(),
        "coarse_dyn_patch": dyn_maps,
        "dyn4d_patch": dyn_maps,
    }


def build_dyn4d_variant_maps(
    global_q_raw_patchvec: torch.Tensor | None,
    global_k_raw_patchvec: torch.Tensor | None,
    global_q_raw_patchvec_layers: torch.Tensor | None = None,
    global_k_raw_patchvec_layers: torch.Tensor | None = None,
    global_layer_ids: torch.Tensor | None = None,
    images: torch.Tensor | None = None,
    world_points: torch.Tensor | None = None,
    local_points: torch.Tensor | None = None,
    camera_poses: torch.Tensor | None = None,
    cue_patch: torch.Tensor | None = None,
) -> List[dict[str, object]]:
    """Build several offline 4D_dyn variants from merged raw global q/k tensors."""
    if global_q_raw_patchvec_layers is not None and global_k_raw_patchvec_layers is not None:
        q_source = global_q_raw_patchvec_layers
        k_source = global_k_raw_patchvec_layers
    else:
        q_source = global_q_raw_patchvec
        k_source = global_k_raw_patchvec

    if q_source is None or k_source is None:
        return []

    variant_specs = [
        {
            "name": "base_r2_35_40_25",
            "title": "Base r2 35/40/25",
            "radius": 2,
            "weights": (0.35, 0.40, 0.25),
        },
        {
            "name": "qk70_r2_15_70_15",
            "title": "QK70 r2 15/70/15",
            "radius": 2,
            "weights": (0.15, 0.70, 0.15),
        },
        {
            "name": "qk70_r4_15_70_15",
            "title": "QK70 r4 15/70/15",
            "radius": 4,
            "weights": (0.15, 0.70, 0.15),
        },
        {
            "name": "qk100_r4_0_100_0",
            "title": "QK only r4",
            "radius": 4,
            "weights": (0.0, 1.0, 0.0),
        },
    ]

    stats_cache: dict[int, dict[str, torch.Tensor]] = {}
    variants: List[dict[str, object]] = []
    if global_q_raw_patchvec_layers is not None and global_k_raw_patchvec_layers is not None:
        variants.append(_build_vggt4d_direct_variant(
            global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers,
        ))
        variants.append(_build_vggt4d_custom_variant(
            global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers,
            global_layer_ids=global_layer_ids,
        ))
        variants.append(_build_loger_lite_variant(
            global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers,
            global_layer_ids=global_layer_ids,
        ))
    for spec in variant_specs:
        radius = int(spec["radius"])
        if radius not in stats_cache:
            stats_cache[radius] = _compute_dyn4d_token_gram_stats(
                q_source,
                k_source,
                window_radius=radius,
            )
        stats = stats_cache[radius]
        variants.append({
            "name": spec["name"],
            "title": spec["title"],
            "radius": radius,
            "weights": spec["weights"],
            "dyn4d_patch": _compose_dyn4d_from_components(
                stats["qq_mean_patch"],
                stats["qk_var_patch"],
                stats["kk_mean_patch"],
                spec["weights"],
            ),
        })

    return variants


def _render_frame_attention_row(
    frame_attention_prior: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
) -> np.ndarray:
    row = frame_attention_prior[frame_idx:frame_idx + 1]
    row = np.repeat(row, 24, axis=0)
    row = cv2.resize(row, (width, height), interpolation=cv2.INTER_NEAREST)
    return _colorize_map(row, cv2.COLORMAP_TURBO, value_range=(0.0, 1.0))


def _render_frame_attention_matrix(
    frame_attention_prior: np.ndarray,
    frame_idx: int,
    height: int,
    width: int,
) -> np.ndarray:
    panel = _colorize_map(
        frame_attention_prior, cv2.COLORMAP_VIRIDIS, value_range=(0.0, 1.0),
    )
    panel = cv2.resize(panel, (width, height), interpolation=cv2.INTER_NEAREST)
    total_frames = max(frame_attention_prior.shape[0], 1)
    x = int(round(frame_idx / max(total_frames - 1, 1) * (width - 1)))
    y = int(round(frame_idx / max(total_frames - 1, 1) * (height - 1)))
    cv2.line(panel, (x, 0), (x, height - 1), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(panel, (0, y), (width - 1, y), (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_frame_attention_summary(
    frame_attention_prior: torch.Tensor,
    output_path: str,
) -> None:
    """Save a single heatmap image for the merged frame-affinity matrix."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mat = frame_attention_prior.cpu().numpy()
    vis = _colorize_map(mat, cv2.COLORMAP_VIRIDIS, value_range=(0.0, 1.0))
    cv2.imwrite(output_path, vis)
    print(f"Saved frame-attention summary to {output_path}")


def create_cue_video(
    images: torch.Tensor,
    cue: CueOutput,
    output_path: str,
    fps: int = 10,
    save_frames_dir: str | None = None,
    attn_dynamic_patch: torch.Tensor | None = None,
    dyn4d_patch: torch.Tensor | None = None,
    dyn4d_qq_mean_patch: torch.Tensor | None = None,
    dyn4d_qk_var_patch: torch.Tensor | None = None,
    dyn4d_kk_mean_patch: torch.Tensor | None = None,
    global_q_raw_patchvec: torch.Tensor | None = None,
    global_k_raw_patchvec: torch.Tensor | None = None,
    frame_attention_prior: torch.Tensor | None = None,
    frame_attn_cosine_shallow: torch.Tensor | None = None,
    frame_attn_cosine_deep: torch.Tensor | None = None,
    frame_attn_cosine_avg: torch.Tensor | None = None,
    frame_attn_key_cosine_shallow: torch.Tensor | None = None,
    frame_attn_key_cosine_deep: torch.Tensor | None = None,
    frame_attn_key_cosine_avg: torch.Tensor | None = None,
) -> None:
    """Create a cue video and, when available, append Stage-A attention priors."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    T, _, H, W = images.shape
    cue_maps = cue.E_cue.permute(0, 3, 1, 2).cpu()  # [T, 5, Hc, Wc]
    cue_maps = F.interpolate(cue_maps, size=(H, W), mode="bilinear", align_corners=False)
    cue_maps_np = cue_maps.numpy()

    c_dyn_explicit_np = None
    if cue.C_dyn_explicit is not None:
        c_dyn_explicit = F.interpolate(
            cue.C_dyn_explicit.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        c_dyn_explicit_np = c_dyn_explicit[:, 0].numpy()

    c_dyn_implicit_np = None
    if cue.C_dyn_implicit is not None:
        c_dyn_implicit = F.interpolate(
            cue.C_dyn_implicit.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        c_dyn_implicit_np = c_dyn_implicit[:, 0].numpy()

    c_dyn_fusion_max_np = None
    if cue.C_dyn_fusion_max is not None:
        c_dyn_fusion_max = F.interpolate(
            cue.C_dyn_fusion_max.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        c_dyn_fusion_max_np = c_dyn_fusion_max[:, 0].numpy()

    c_dyn_fusion_soft_or_np = None
    if cue.C_dyn_fusion_soft_or is not None:
        c_dyn_fusion_soft_or = F.interpolate(
            cue.C_dyn_fusion_soft_or.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        c_dyn_fusion_soft_or_np = c_dyn_fusion_soft_or[:, 0].numpy()

    c_dyn_fusion_avg_np = None
    if cue.C_dyn_fusion_avg is not None:
        c_dyn_fusion_avg = F.interpolate(
            cue.C_dyn_fusion_avg.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        c_dyn_fusion_avg_np = c_dyn_fusion_avg[:, 0].numpy()

    c_dyn_fusion_addclip_np = None
    if cue.C_dyn_fusion_addclip is not None:
        c_dyn_fusion_addclip = F.interpolate(
            cue.C_dyn_fusion_addclip.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        c_dyn_fusion_addclip_np = c_dyn_fusion_addclip[:, 0].numpy()

    g_write = cue.G_write_geo.unsqueeze(1).cpu()  # [T, 1, Hc, Wc]
    g_write = F.interpolate(g_write, size=(H, W), mode="bilinear", align_corners=False)
    g_write_np = g_write[:, 0].numpy()

    attn_dynamic_np = None
    if attn_dynamic_patch is not None:
        attn_dynamic = F.interpolate(
            attn_dynamic_patch.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        attn_dynamic_np = attn_dynamic[:, 0].numpy()

    dyn4d_np = None
    if dyn4d_patch is not None:
        dyn4d = F.interpolate(
            dyn4d_patch.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        dyn4d_np = dyn4d[:, 0].numpy()

    dyn4d_qq_mean_np = None
    if dyn4d_qq_mean_patch is not None:
        dyn4d_qq_mean = F.interpolate(
            dyn4d_qq_mean_patch.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        dyn4d_qq_mean_np = dyn4d_qq_mean[:, 0].numpy()

    dyn4d_qk_var_np = None
    if dyn4d_qk_var_patch is not None:
        dyn4d_qk_var = F.interpolate(
            dyn4d_qk_var_patch.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        dyn4d_qk_var_np = dyn4d_qk_var[:, 0].numpy()

    dyn4d_kk_mean_np = None
    if dyn4d_kk_mean_patch is not None:
        dyn4d_kk_mean = F.interpolate(
            dyn4d_kk_mean_patch.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        dyn4d_kk_mean_np = dyn4d_kk_mean[:, 0].numpy()

    global_q_raw_np = None
    if global_q_raw_patchvec is not None:
        global_q_raw = _summarize_patchvec_response(global_q_raw_patchvec.cpu())
        global_q_raw = F.interpolate(
            global_q_raw.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        global_q_raw_np = global_q_raw[:, 0].numpy()

    global_k_raw_np = None
    if global_k_raw_patchvec is not None:
        global_k_raw = _summarize_patchvec_response(global_k_raw_patchvec.cpu())
        global_k_raw = F.interpolate(
            global_k_raw.unsqueeze(1),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        global_k_raw_np = global_k_raw[:, 0].numpy()

    frame_attention_np = (
        frame_attention_prior.cpu().numpy() if frame_attention_prior is not None else None
    )

    frame_attn_cosine_shallow_np = None
    if frame_attn_cosine_shallow is not None:
        frame_attn_cosine_shallow_np = F.interpolate(
            frame_attn_cosine_shallow.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    frame_attn_cosine_deep_np = None
    if frame_attn_cosine_deep is not None:
        frame_attn_cosine_deep_np = F.interpolate(
            frame_attn_cosine_deep.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    frame_attn_cosine_avg_np = None
    if frame_attn_cosine_avg is not None:
        frame_attn_cosine_avg_np = F.interpolate(
            frame_attn_cosine_avg.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    frame_attn_key_cosine_shallow_np = None
    if frame_attn_key_cosine_shallow is not None:
        frame_attn_key_cosine_shallow_np = F.interpolate(
            frame_attn_key_cosine_shallow.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    frame_attn_key_cosine_deep_np = None
    if frame_attn_key_cosine_deep is not None:
        frame_attn_key_cosine_deep_np = F.interpolate(
            frame_attn_key_cosine_deep.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    frame_attn_key_cosine_avg_np = None
    if frame_attn_key_cosine_avg is not None:
        frame_attn_key_cosine_avg_np = F.interpolate(
            frame_attn_key_cosine_avg.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()

    has_stagea_prior_row = attn_dynamic_np is not None or frame_attention_np is not None
    has_dyn4d_row = any(
        x is not None for x in (
            dyn4d_np,
            dyn4d_qq_mean_np,
            dyn4d_qk_var_np,
            dyn4d_kk_mean_np,
        )
    )
    has_global_qk_row = any(
        x is not None for x in (
            global_q_raw_np,
            global_k_raw_np,
        )
    )
    has_mut3r_query_row = any(
        x is not None for x in (
            frame_attn_cosine_shallow_np,
            frame_attn_cosine_deep_np,
            frame_attn_cosine_avg_np,
        )
    )
    has_mut3r_key_row = any(
        x is not None for x in (
            frame_attn_key_cosine_shallow_np,
            frame_attn_key_cosine_deep_np,
            frame_attn_key_cosine_avg_np,
        )
    )
    has_dyn_branch_row = c_dyn_explicit_np is not None or c_dyn_implicit_np is not None
    has_fusion_compare_row = any(
        x is not None for x in (
            c_dyn_fusion_max_np,
            c_dyn_fusion_soft_or_np,
            c_dyn_fusion_avg_np,
            c_dyn_fusion_addclip_np,
        )
    )
    num_rows = (
        2
        + int(has_dyn_branch_row)
        + int(has_fusion_compare_row)
        + int(has_global_qk_row)
        + int(has_dyn4d_row)
        + int(has_stagea_prior_row)
        + int(has_mut3r_query_row)
        + int(has_mut3r_key_row)
    )

    gap = 4
    grid_w = W * 4 + gap * 3
    grid_h = H * num_rows + gap * (num_rows - 1)
    colorbar_w = 96 + gap
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w + colorbar_w, grid_h),
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

        rows = [top, bottom]

        if has_dyn_branch_row:
            blank = np.zeros((H, W, 3), np.uint8)
            explicit_vis = _render_panel("C_dyn explicit", blank)
            explicit_overlay = _render_panel("Explicit overlay", blank)
            if c_dyn_explicit_np is not None:
                explicit_color = _colorize_map(
                    c_dyn_explicit_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0),
                )
                explicit_vis = _render_panel("C_dyn explicit", explicit_color)
                explicit_overlay = _render_panel(
                    "Explicit overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, explicit_color, 0.55, 0),
                )

            implicit_vis = _render_panel("C_dyn implicit", blank)
            implicit_overlay = _render_panel("Implicit overlay", blank)
            if c_dyn_implicit_np is not None:
                implicit_color = _colorize_map(
                    c_dyn_implicit_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0),
                )
                implicit_vis = _render_panel("C_dyn implicit", implicit_color)
                implicit_overlay = _render_panel(
                    "Implicit overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, implicit_color, 0.55, 0),
                )

            dyn_branch_row = np.concatenate([
                explicit_vis, np.zeros((H, gap, 3), np.uint8),
                explicit_overlay, np.zeros((H, gap, 3), np.uint8),
                implicit_vis, np.zeros((H, gap, 3), np.uint8),
                implicit_overlay,
            ], axis=1)
            rows.append(dyn_branch_row)

        if has_fusion_compare_row:
            blank = np.zeros((H, W, 3), np.uint8)
            fuse_max_vis = _render_panel("Fuse max", blank)
            if c_dyn_fusion_max_np is not None:
                fuse_max_vis = _render_panel(
                    "Fuse max",
                    _colorize_map(c_dyn_fusion_max_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )

            fuse_soft_vis = _render_panel("Fuse soft-or", blank)
            if c_dyn_fusion_soft_or_np is not None:
                fuse_soft_vis = _render_panel(
                    "Fuse soft-or",
                    _colorize_map(c_dyn_fusion_soft_or_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )

            fuse_avg_vis = _render_panel("Fuse avg", blank)
            if c_dyn_fusion_avg_np is not None:
                fuse_avg_vis = _render_panel(
                    "Fuse avg",
                    _colorize_map(c_dyn_fusion_avg_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )

            fuse_add_vis = _render_panel("Fuse addclip", blank)
            if c_dyn_fusion_addclip_np is not None:
                fuse_add_vis = _render_panel(
                    "Fuse addclip",
                    _colorize_map(c_dyn_fusion_addclip_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )

            fusion_row = np.concatenate([
                fuse_max_vis, np.zeros((H, gap, 3), np.uint8),
                fuse_soft_vis, np.zeros((H, gap, 3), np.uint8),
                fuse_avg_vis, np.zeros((H, gap, 3), np.uint8),
                fuse_add_vis,
            ], axis=1)
            rows.append(fusion_row)

        if has_global_qk_row:
            blank = np.zeros((H, W, 3), np.uint8)
            global_q_vis = _render_panel("Global q raw", blank)
            global_q_overlay = _render_panel("Global q overlay", blank)
            global_k_vis = _render_panel("Global k raw", blank)
            global_k_overlay = _render_panel("Global k overlay", blank)

            if global_q_raw_np is not None:
                global_q_color = _colorize_map(
                    global_q_raw_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0),
                )
                global_q_vis = _render_panel("Global q raw", global_q_color)
                global_q_overlay = _render_panel(
                    "Global q overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, global_q_color, 0.55, 0),
                )

            if global_k_raw_np is not None:
                global_k_color = _colorize_map(
                    global_k_raw_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0),
                )
                global_k_vis = _render_panel("Global k raw", global_k_color)
                global_k_overlay = _render_panel(
                    "Global k overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, global_k_color, 0.55, 0),
                )

            global_qk_row = np.concatenate([
                global_q_vis, np.zeros((H, gap, 3), np.uint8),
                global_q_overlay, np.zeros((H, gap, 3), np.uint8),
                global_k_vis, np.zeros((H, gap, 3), np.uint8),
                global_k_overlay,
            ], axis=1)
            rows.append(global_qk_row)

        if has_dyn4d_row:
            blank = np.zeros((H, W, 3), np.uint8)
            qq_mean_vis = _render_panel("4D qq_mean", blank)
            qk_var_vis = _render_panel("4D qk_var", blank)
            kk_mean_vis = _render_panel("4D kk_mean", blank)
            dyn4d_vis = _render_panel("4D_dyn", blank)

            if dyn4d_qq_mean_np is not None:
                qq_mean_vis = _render_panel(
                    "4D qq_mean",
                    _colorize_map(dyn4d_qq_mean_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )
            if dyn4d_qk_var_np is not None:
                qk_var_vis = _render_panel(
                    "4D qk_var",
                    _colorize_map(dyn4d_qk_var_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )
            if dyn4d_kk_mean_np is not None:
                kk_mean_vis = _render_panel(
                    "4D kk_mean",
                    _colorize_map(dyn4d_kk_mean_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )
            if dyn4d_np is not None:
                dyn4d_vis = _render_panel(
                    "4D_dyn",
                    _colorize_map(dyn4d_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )

            dyn4d_row = np.concatenate([
                qq_mean_vis, np.zeros((H, gap, 3), np.uint8),
                qk_var_vis, np.zeros((H, gap, 3), np.uint8),
                kk_mean_vis, np.zeros((H, gap, 3), np.uint8),
                dyn4d_vis,
            ], axis=1)
            rows.append(dyn4d_row)

        if has_stagea_prior_row:
            blank = np.zeros((H, W, 3), np.uint8)
            attn_dyn_vis = _render_panel("StageA attn feat", blank)
            attn_dyn_overlay = _render_panel("StageA feat overlay", blank)
            if attn_dynamic_np is not None:
                attn_dyn_color = _colorize_map(
                    attn_dynamic_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0),
                )
                attn_dyn_vis = _render_panel("StageA attn feat", attn_dyn_color)
                attn_dyn_overlay = _render_panel(
                    "StageA feat overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, attn_dyn_color, 0.55, 0),
                )

            frame_row_vis = blank
            frame_mat_vis = blank
            if frame_attention_np is not None:
                frame_row_vis = _render_panel(
                    "StageA frame-attn row",
                    _render_frame_attention_row(frame_attention_np, t, H, W),
                )
                frame_mat_vis = _render_panel(
                    "StageA frame-attn mat",
                    _render_frame_attention_matrix(frame_attention_np, t, H, W),
                )
            else:
                frame_row_vis = _render_panel("StageA frame-attn row", blank)
                frame_mat_vis = _render_panel("StageA frame-attn mat", blank)

            extra = np.concatenate([
                attn_dyn_vis, np.zeros((H, gap, 3), np.uint8),
                attn_dyn_overlay, np.zeros((H, gap, 3), np.uint8),
                frame_row_vis, np.zeros((H, gap, 3), np.uint8),
                frame_mat_vis,
            ], axis=1)
            rows.append(extra)

        if has_mut3r_query_row:
            blank = np.zeros((H, W, 3), np.uint8)

            mut3r_shallow_vis = _render_panel("MUT3R q cos shallow", blank)
            if frame_attn_cosine_shallow_np is not None:
                mut3r_shallow_vis = _render_panel(
                    "MUT3R q cos shallow",
                    _colorize_map(
                        frame_attn_cosine_shallow_np[t],
                        cv2.COLORMAP_TURBO,
                        value_range=(0.0, 1.0),
                    ),
                )

            mut3r_deep_vis = _render_panel("MUT3R q cos deep", blank)
            if frame_attn_cosine_deep_np is not None:
                mut3r_deep_vis = _render_panel(
                    "MUT3R q cos deep",
                    _colorize_map(
                        frame_attn_cosine_deep_np[t],
                        cv2.COLORMAP_TURBO,
                        value_range=(0.0, 1.0),
                    ),
                )

            mut3r_avg_vis = _render_panel("MUT3R q cos avg", blank)
            mut3r_avg_overlay = _render_panel("MUT3R q avg overlay", blank)
            if frame_attn_cosine_avg_np is not None:
                mut3r_avg_color = _colorize_map(
                    frame_attn_cosine_avg_np[t],
                    cv2.COLORMAP_TURBO,
                    value_range=(0.0, 1.0),
                )
                mut3r_avg_vis = _render_panel("MUT3R q cos avg", mut3r_avg_color)
                mut3r_avg_overlay = _render_panel(
                    "MUT3R q avg overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, mut3r_avg_color, 0.55, 0),
                )

            mut3r_row = np.concatenate([
                mut3r_shallow_vis, np.zeros((H, gap, 3), np.uint8),
                mut3r_deep_vis, np.zeros((H, gap, 3), np.uint8),
                mut3r_avg_vis, np.zeros((H, gap, 3), np.uint8),
                mut3r_avg_overlay,
            ], axis=1)
            rows.append(mut3r_row)

        if has_mut3r_key_row:
            blank = np.zeros((H, W, 3), np.uint8)

            key_shallow_vis = _render_panel("MUT3R k cos shallow", blank)
            if frame_attn_key_cosine_shallow_np is not None:
                key_shallow_vis = _render_panel(
                    "MUT3R k cos shallow",
                    _colorize_map(
                        frame_attn_key_cosine_shallow_np[t],
                        cv2.COLORMAP_TURBO,
                        value_range=(0.0, 1.0),
                    ),
                )

            key_deep_vis = _render_panel("MUT3R k cos deep", blank)
            if frame_attn_key_cosine_deep_np is not None:
                key_deep_vis = _render_panel(
                    "MUT3R k cos deep",
                    _colorize_map(
                        frame_attn_key_cosine_deep_np[t],
                        cv2.COLORMAP_TURBO,
                        value_range=(0.0, 1.0),
                    ),
                )

            key_avg_vis = _render_panel("MUT3R k cos avg", blank)
            key_avg_overlay = _render_panel("MUT3R k avg overlay", blank)
            if frame_attn_key_cosine_avg_np is not None:
                key_avg_color = _colorize_map(
                    frame_attn_key_cosine_avg_np[t],
                    cv2.COLORMAP_TURBO,
                    value_range=(0.0, 1.0),
                )
                key_avg_vis = _render_panel("MUT3R k cos avg", key_avg_color)
                key_avg_overlay = _render_panel(
                    "MUT3R k avg overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, key_avg_color, 0.55, 0),
                )

            key_row = np.concatenate([
                key_shallow_vis, np.zeros((H, gap, 3), np.uint8),
                key_deep_vis, np.zeros((H, gap, 3), np.uint8),
                key_avg_vis, np.zeros((H, gap, 3), np.uint8),
                key_avg_overlay,
            ], axis=1)
            rows.append(key_row)

        frame = rows[0]
        for row in rows[1:]:
            frame = np.concatenate([frame, np.zeros((gap, grid_w, 3), np.uint8), row], axis=0)
        frame = _append_colorbar_legend(frame, cv2.COLORMAP_TURBO, title="Value")

        writer.write(frame)
        if save_frames_dir:
            cv2.imwrite(os.path.join(save_frames_dir, f"frame_{t:05d}.jpg"), frame)

    writer.release()
    print(f"Saved cue visualisation video to {output_path}  ({T} frames, {fps} FPS)")


def create_dyn4d_variant_video(
    images: torch.Tensor,
    variant_maps: List[dict[str, object]],
    output_path: str,
    fps: int,
) -> None:
    """Create an extra comparison video for several offline 4D_dyn variants."""
    if not variant_maps:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    T, _, H, W = images.shape
    gap = 4
    num_cols = len(variant_maps)
    grid_w = W * num_cols + gap * max(num_cols - 1, 0)
    grid_h = H * 2 + gap
    colorbar_w = 96 + gap

    prepared: List[Tuple[str, np.ndarray]] = []
    for variant in variant_maps:
        dyn4d_patch = variant["dyn4d_patch"]
        if not isinstance(dyn4d_patch, torch.Tensor):
            continue
        dyn4d_up = F.interpolate(
            dyn4d_patch.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        prepared.append((str(variant["title"]), dyn4d_up))

    if not prepared:
        return

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w + colorbar_w, grid_h),
    )

    for t in range(T):
        rgb = (images[t].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        top_panels: List[np.ndarray] = []
        bottom_panels: List[np.ndarray] = []
        for title, dyn_np in prepared:
            dyn_color = _colorize_map(dyn_np[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0))
            top_panels.append(_render_panel(title, dyn_color))
            bottom_panels.append(
                _render_panel(
                    f"{title} overlay",
                    cv2.addWeighted(rgb_bgr, 0.45, dyn_color, 0.55, 0),
                )
            )

        top = top_panels[0]
        for panel in top_panels[1:]:
            top = np.concatenate([top, np.zeros((H, gap, 3), np.uint8), panel], axis=1)

        bottom = bottom_panels[0]
        for panel in bottom_panels[1:]:
            bottom = np.concatenate([bottom, np.zeros((H, gap, 3), np.uint8), panel], axis=1)

        frame = np.concatenate([top, np.zeros((gap, grid_w, 3), np.uint8), bottom], axis=0)
        frame = _append_colorbar_legend(frame, cv2.COLORMAP_TURBO, title="Value")
        writer.write(frame)

    writer.release()
    print(f"Saved 4D_dyn comparison video to {output_path}  ({T} frames, {fps} FPS)")


def create_vggt4d_terms_video(
    images: torch.Tensor,
    variant: dict[str, object],
    output_path: str,
    fps: int,
) -> None:
    """Create a diagnostic video for VGGT4D-direct intermediate terms."""
    required_keys = [
        "mean1_patch",
        "var1_patch",
        "mean2_patch",
        "mean3_patch",
        "var3_patch",
        "dyn4d_patch",
    ]
    if any(key not in variant or not isinstance(variant[key], torch.Tensor) for key in required_keys):
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    T, _, H, W = images.shape
    gap = 4
    num_cols = 3
    num_rows = 2
    grid_w = W * num_cols + gap * (num_cols - 1)
    grid_h = H * num_rows + gap * (num_rows - 1)
    colorbar_w = 96 + gap

    panels = [
        ("mean1", variant["mean1_patch"]),
        ("var1", variant["var1_patch"]),
        ("mean2", variant["mean2_patch"]),
        ("mean3", variant["mean3_patch"]),
        ("var3", variant["var3_patch"]),
        ("dyn", variant["dyn4d_patch"]),
    ]
    prepared: List[Tuple[str, np.ndarray]] = []
    for title, patch_map in panels:
        up = F.interpolate(
            patch_map.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        prepared.append((title, up))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w + colorbar_w, grid_h),
    )

    for t in range(T):
        row_imgs = []
        for row_idx in range(num_rows):
            row_panels = []
            for col_idx in range(num_cols):
                idx = row_idx * num_cols + col_idx
                title, panel_map = prepared[idx]
                panel = _render_panel(
                    f"VGGT4D {title}",
                    _colorize_map(panel_map[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )
                row_panels.append(panel)
                if col_idx < num_cols - 1:
                    row_panels.append(np.zeros((H, gap, 3), np.uint8))
            row_imgs.append(np.concatenate(row_panels, axis=1))

        frame = row_imgs[0]
        for row in row_imgs[1:]:
            frame = np.concatenate([frame, np.zeros((gap, grid_w, 3), np.uint8), row], axis=0)
        frame = _append_colorbar_legend(frame, cv2.COLORMAP_TURBO, title="Value")
        writer.write(frame)

    writer.release()
    print(f"Saved VGGT4D-direct terms video to {output_path}  ({T} frames, {fps} FPS)")


def create_vggt4d_custom_terms_video(
    images: torch.Tensor,
    variant: dict[str, object],
    output_path: str,
    fps: int,
) -> None:
    """Create a diagnostic video for VGGT4D-custom terms."""
    required_keys = [
        "shallow_term_patch",
        "middle_term_patch",
        "deep_term_patch",
        "dyn4d_patch",
    ]
    if any(key not in variant or not isinstance(variant[key], torch.Tensor) for key in required_keys):
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    T, _, H, W = images.shape
    gap = 4
    num_cols = 4
    panels = [("shallow", variant["shallow_term_patch"])]
    panels.append(("middle", variant["middle_term_patch"]))
    panels.append(("deep", variant["deep_term_patch"]))
    if isinstance(variant.get("coarse_dyn_patch"), torch.Tensor):
        panels.append(("coarse", variant["coarse_dyn_patch"]))
    if isinstance(variant.get("cluster_dyn_patch"), torch.Tensor):
        panels.append(("cluster", variant["cluster_dyn_patch"]))
    if isinstance(variant.get("binary_mask_patch"), torch.Tensor):
        panels.append(("mask", variant["binary_mask_patch"]))
    panels.append(("dyn", variant["dyn4d_patch"]))

    num_rows = (len(panels) + num_cols - 1) // num_cols
    grid_w = W * num_cols + gap * (num_cols - 1)
    grid_h = H * num_rows + gap * (num_rows - 1)
    colorbar_w = 96 + gap

    prepared: List[Tuple[str, np.ndarray]] = []
    for title, patch_map in panels:
        up = F.interpolate(
            patch_map.unsqueeze(1).cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        prepared.append((title, up))

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w + colorbar_w, grid_h),
    )

    for t in range(T):
        row_imgs = []
        for row_idx in range(num_rows):
            row_panels = []
            for col_idx in range(num_cols):
                idx = row_idx * num_cols + col_idx
                if idx < len(prepared):
                    title, panel_map = prepared[idx]
                    panel = _render_panel(
                        f"VGGT4D-custom {title}",
                        _colorize_map(panel_map[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                    )
                else:
                    panel = np.zeros((H, W, 3), np.uint8)
                row_panels.append(panel)
                if col_idx < num_cols - 1:
                    row_panels.append(np.zeros((H, gap, 3), np.uint8))
            row_imgs.append(np.concatenate(row_panels, axis=1))

        frame = row_imgs[0]
        for row in row_imgs[1:]:
            frame = np.concatenate([frame, np.zeros((gap, grid_w, 3), np.uint8), row], axis=0)
        frame = _append_colorbar_legend(frame, cv2.COLORMAP_TURBO, title="Value")
        writer.write(frame)

    writer.release()
    print(f"Saved VGGT4D-custom terms video to {output_path}  ({T} frames, {fps} FPS)")


def create_attention_layer_grid_video(
    layer_maps: torch.Tensor,
    layer_ids: torch.Tensor,
    output_path: str,
    fps: int,
    prefix: str,
    vis_mode: str = "fixed",
) -> None:
    """Create a per-frame grid video for all frame-attention layers."""
    if layer_maps is None or layer_ids is None:
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    maps_np = layer_maps.cpu().numpy()
    layer_ids_np = layer_ids.cpu().numpy().astype(np.int32)

    T, L, Hm, Wm = maps_np.shape
    if L == 0:
        return

    num_cols = 6
    num_rows = int(np.ceil(L / num_cols))
    gap = 4
    max_grid_width = 1920
    cell_w = min(256, max(120, (max_grid_width - gap * (num_cols - 1)) // num_cols))
    cell_h = max(96, int(round(cell_w * Hm / max(Wm, 1))))
    grid_w = num_cols * cell_w + gap * (num_cols - 1)
    grid_h = num_rows * cell_h + gap * (num_rows - 1)
    colorbar_w = 96 + gap

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w + colorbar_w, grid_h),
    )

    blank = np.zeros((cell_h, cell_w, 3), np.uint8)
    for t in range(T):
        panels = []
        for layer_idx in range(L):
            panel_map, value_range = _prepare_layer_vis_map(
                maps_np[t, layer_idx],
                mode=vis_mode,
            )
            panel = _colorize_map(
                panel_map,
                cv2.COLORMAP_TURBO,
                value_range=value_range,
            )
            panel = cv2.resize(panel, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            panel = _render_panel(f"{prefix} L{int(layer_ids_np[layer_idx])}", panel)
            panels.append(panel)

        while len(panels) < num_cols * num_rows:
            panels.append(_render_panel(f"{prefix} --", blank))

        row_imgs = []
        for row_idx in range(num_rows):
            row_panels = []
            for col_idx in range(num_cols):
                panel = panels[row_idx * num_cols + col_idx]
                row_panels.append(panel)
                if col_idx < num_cols - 1:
                    row_panels.append(np.zeros((cell_h, gap, 3), np.uint8))
            row_imgs.append(np.concatenate(row_panels, axis=1))

        frame = row_imgs[0]
        for row in row_imgs[1:]:
            frame = np.concatenate([frame, np.zeros((gap, grid_w, 3), np.uint8), row], axis=0)
        legend_title = "Contrast" if vis_mode == "contrast" else "Value"
        frame = _append_colorbar_legend(frame, cv2.COLORMAP_TURBO, title=legend_title)

        writer.write(frame)

    writer.release()
    print(f"Saved {prefix} layer-grid video to {output_path}  ({T} frames, {fps} FPS)")


def _parse_offset_list(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    offsets = sorted({int(p) for p in parts})
    offsets = [o for o in offsets if o != 0]
    if not offsets:
        offsets = [-6, -4, -2, 2, 4, 6]
    return offsets


def _resolve_target_layer_ids(total_layers: int, spec: tuple[int, int]) -> torch.Tensor:
    start, end = spec
    return _map_vggt4d_layer_range(total_layers, start, end)


def _extract_family_map_from_qk_with_offsets(
    ref_id: int,
    q_layers: torch.Tensor,
    k_layers: torch.Tensor,
    patch_hw: tuple[int, int],
    layer_ids: torch.Tensor,
    offsets: List[int],
    mode: str,
) -> torch.Tensor:
    """VGGT4D-style map extraction on headed q/k tensors.

    q_layers / k_layers: [T, L, H_head, N_tok, D]
    """
    T = q_layers.shape[0]
    H_tok, W_tok = patch_hw
    src_ids = [ref_id + o for o in offsets if 0 <= ref_id + o < T]
    if not src_ids or layer_ids.numel() == 0:
        return torch.zeros(H_tok, W_tok, device=q_layers.device, dtype=torch.float32)

    q_ref = q_layers[ref_id, layer_ids]                 # [L, Hh, N, D]
    k_ref = k_layers[ref_id, layer_ids]                 # [L, Hh, N, D]
    q_src = q_layers[src_ids][:, layer_ids]             # [S, L, Hh, N, D]
    k_src = k_layers[src_ids][:, layer_ids]             # [S, L, Hh, N, D]

    if mode in ("mean1", "mean2", "qq_mean"):
        scores = torch.einsum("l h q d, s l h k d -> s l h q k", q_ref, q_src)
        map_q = scores.mean(dim=(0, 1, 2, 4))          # [Q]
    elif mode in ("var1", "qq_var"):
        scores = torch.einsum("l h q d, s l h k d -> s l h q k", q_ref, q_src)
        map_q = scores.mean(dim=(0, 1, 2)).std(dim=-1)  # [Q]
    elif mode in ("mean3", "kk_mean"):
        scores = torch.einsum("l h q d, s l h k d -> s l h q k", k_ref, k_src)
        map_q = scores.mean(dim=(0, 1, 2, 4))
    elif mode in ("var3", "qk_var"):
        scores = torch.einsum("l h q d, s l h k d -> s l h q k", q_ref, k_src)
        map_q = scores.mean(dim=(0, 1, 2)).std(dim=-1)
    elif mode == "kq_var":
        scores = torch.einsum("l h q d, s l h k d -> s l h q k", k_ref, q_src)
        map_q = scores.mean(dim=(0, 1, 2)).std(dim=-1)
    else:
        raise ValueError(f"Unsupported family mode: {mode}")

    map_2d = map_q.reshape(H_tok, W_tok)
    return _normalize_map_2d_tensor(map_2d)


def _compute_vggt4d_family_targets(
    global_q: torch.Tensor,
    global_k: torch.Tensor,
    patch_hw: tuple[int, int],
    offsets: List[int],
) -> dict[str, torch.Tensor]:
    """Compute VGGT4D reference family maps and final dyn map.

    global_q/global_k: [T, L, H_head, N_tok, D]
    """
    T, L = global_q.shape[:2]
    specs = {
        "mean1": (3, 8),
        "var1": (18, 20),
        "mean2": (17, 22),
        "mean3": (0, 1),
        "var3": (0, 1),
        "kq_var_diag": (0, 1),
    }
    layer_ids = {k: _resolve_target_layer_ids(L, v).to(global_q.device) for k, v in specs.items()}

    out: Dict[str, List[torch.Tensor]] = {
        "mean1": [],
        "var1": [],
        "mean2": [],
        "mean3": [],
        "var3": [],
        "kq_var_diag": [],
        "dyn": [],
    }
    for t in range(T):
        mean1 = _extract_family_map_from_qk_with_offsets(t, global_q, global_k, patch_hw, layer_ids["mean1"], offsets, "mean1")
        var1 = _extract_family_map_from_qk_with_offsets(t, global_q, global_k, patch_hw, layer_ids["var1"], offsets, "var1")
        mean2 = _extract_family_map_from_qk_with_offsets(t, global_q, global_k, patch_hw, layer_ids["mean2"], offsets, "mean2")
        mean3 = _extract_family_map_from_qk_with_offsets(t, global_q, global_k, patch_hw, layer_ids["mean3"], offsets, "mean3")
        var3 = _extract_family_map_from_qk_with_offsets(t, global_q, global_k, patch_hw, layer_ids["var3"], offsets, "var3")
        kq_var = _extract_family_map_from_qk_with_offsets(t, global_q, global_k, patch_hw, layer_ids["kq_var_diag"], offsets, "kq_var")
        dyn = _normalize_map_2d_tensor((1.0 - mean1) * (1.0 - var1) * mean2 * (1.0 - mean3) * var3)
        out["mean1"].append(mean1)
        out["var1"].append(var1)
        out["mean2"].append(mean2)
        out["mean3"].append(mean3)
        out["var3"].append(var3)
        out["kq_var_diag"].append(kq_var)
        out["dyn"].append(dyn)

    return {
        key: torch.stack(val, dim=0).detach().cpu()
        for key, val in out.items()
    }


def _compute_loger_family_candidates_with_offsets(
    q_layers: torch.Tensor,
    k_layers: torch.Tensor,
    offsets: List[int],
) -> dict[str, torch.Tensor]:
    """Compute per-layer family candidates from LoGeR raw global q/k.

    q_layers/k_layers: [T, L, H_tok, W_tok, D]
    Returns maps [T, L, H_tok, W_tok].
    """
    if q_layers.shape != k_layers.shape:
        raise ValueError("LoGeR raw q/k layer stacks shape mismatch.")
    T, L, H_tok, W_tok, D = q_layers.shape
    N = H_tok * W_tok
    q = F.normalize(q_layers.reshape(T, L, N, D).float(), dim=-1)
    k = F.normalize(k_layers.reshape(T, L, N, D).float(), dim=-1)

    qq_mean = torch.zeros(T, L, N, dtype=torch.float32)
    qq_var = torch.zeros_like(qq_mean)
    kk_mean = torch.zeros_like(qq_mean)
    qk_var = torch.zeros_like(qq_mean)
    kq_var = torch.zeros_like(qq_mean)

    for t in range(T):
        src_ids = [t + o for o in offsets if 0 <= t + o < T]
        if not src_ids:
            continue
        q_ref = q[t]          # [L,N,D]
        k_ref = k[t]
        q_src = q[src_ids]    # [S,L,N,D]
        k_src = k[src_ids]

        qq_scores = torch.einsum("l q d, s l k d -> s l q k", q_ref, q_src)
        kk_scores = torch.einsum("l q d, s l k d -> s l q k", k_ref, k_src)
        qk_scores = torch.einsum("l q d, s l k d -> s l q k", q_ref, k_src)
        kq_scores = torch.einsum("l q d, s l k d -> s l q k", k_ref, q_src)

        qq_mean[t] = ((qq_scores.mean(dim=(0, 3)) + 1.0) * 0.5).clamp(0.0, 1.0)
        kk_mean[t] = ((kk_scores.mean(dim=(0, 3)) + 1.0) * 0.5).clamp(0.0, 1.0)
        qq_var[t] = qq_scores.mean(dim=0).std(dim=-1)
        qk_var[t] = qk_scores.mean(dim=0).std(dim=-1)
        kq_var[t] = kq_scores.mean(dim=0).std(dim=-1)

    def _norm_var_map(var_map: torch.Tensor) -> torch.Tensor:
        flat = var_map.permute(1, 0, 2).reshape(L, -1)
        vmin = flat.amin(dim=-1, keepdim=True)
        vmax = flat.amax(dim=-1, keepdim=True)
        norm = (flat - vmin) / (vmax - vmin).clamp_min(1e-6)
        return norm.reshape(L, T, N).permute(1, 0, 2).reshape(T, L, H_tok, W_tok).clamp(0.0, 1.0)

    return {
        "qq_mean_layers": qq_mean.reshape(T, L, H_tok, W_tok).clamp(0.0, 1.0).cpu(),
        "qq_var_layers": _norm_var_map(qq_var).cpu(),
        "kk_mean_layers": kk_mean.reshape(T, L, H_tok, W_tok).clamp(0.0, 1.0).cpu(),
        "qk_var_layers": _norm_var_map(qk_var).cpu(),
        "kq_var_layers": _norm_var_map(kq_var).cpu(),
    }


def _resize_seq_map_to_hw(seq_map: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    if seq_map.shape[-2:] == hw:
        return seq_map
    return F.interpolate(
        seq_map.unsqueeze(1).float(),
        size=hw,
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-6)
    return float((a @ b) / denom)


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    denom = (a.norm() * b.norm()).clamp_min(1e-6)
    return float((a @ b) / denom)


def _top_percent_iou(a: torch.Tensor, b: torch.Tensor, keep_ratio: float = 0.2) -> float:
    T = a.shape[0]
    ious = []
    for t in range(T):
        aa = a[t].reshape(-1).float()
        bb = b[t].reshape(-1).float()
        qa = torch.quantile(aa, max(0.0, 1.0 - keep_ratio))
        qb = torch.quantile(bb, max(0.0, 1.0 - keep_ratio))
        ma = aa >= qa
        mb = bb >= qb
        inter = (ma & mb).sum().float()
        union = (ma | mb).sum().float().clamp_min(1.0)
        ious.append(float(inter / union))
    return float(np.mean(ious)) if ious else 0.0


def _compute_similarity_metrics(candidate: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    cand = candidate.float()
    tgt = target.float()
    mae = float(torch.mean(torch.abs(cand - tgt)))
    rmse = float(torch.sqrt(torch.mean((cand - tgt) ** 2)))
    pearson = _pearson_corr(cand, tgt)
    cosine = _cosine_sim(cand, tgt)
    iou20 = _top_percent_iou(cand, tgt, keep_ratio=0.2)
    score = 0.50 * pearson + 0.25 * cosine + 0.25 * iou20
    return {
        "score": float(score),
        "pearson": float(pearson),
        "cosine": float(cosine),
        "iou20": float(iou20),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def _auto_select_layers_for_family(
    family_name: str,
    candidate_layers: torch.Tensor,
    target_map: torch.Tensor,
    layer_ids: torch.Tensor,
    topk: int,
) -> dict[str, object]:
    """Return selected weighted family map and diagnostics."""
    T, L, Hc, Wc = candidate_layers.shape
    target_hw = tuple(target_map.shape[-2:])
    if (Hc, Wc) != target_hw:
        cand = _resize_seq_map_to_hw(
            candidate_layers.permute(1, 0, 2, 3).reshape(L * T, Hc, Wc),
            target_hw,
        ).reshape(L, T, *target_hw).permute(1, 0, 2, 3).contiguous()
    else:
        cand = candidate_layers

    layer_reports = []
    score_vec = torch.zeros(L, dtype=torch.float32)
    oriented_layers = []
    for li in range(L):
        layer_map = cand[:, li]
        direct = _compute_similarity_metrics(layer_map, target_map)
        inv = _compute_similarity_metrics(1.0 - layer_map, target_map)
        use_inv = inv["score"] > direct["score"]
        best = inv if use_inv else direct
        oriented = (1.0 - layer_map) if use_inv else layer_map
        oriented_layers.append(oriented)
        score_vec[li] = float(best["score"])
        layer_reports.append({
            "layer_id": int(layer_ids[li].item()),
            "index": int(li),
            "invert": bool(use_inv),
            "metrics": best,
            "metrics_direct": direct,
            "metrics_inverted": inv,
        })

    order = torch.argsort(score_vec, descending=True)
    k = max(1, min(int(topk), L))
    selected_idx = order[:k]
    selected_scores = score_vec[selected_idx]
    weights = torch.softmax(selected_scores * 8.0, dim=0)
    selected_maps = torch.stack([oriented_layers[int(i.item())] for i in selected_idx], dim=0)
    agg_map = torch.sum(weights[:, None, None, None] * selected_maps, dim=0)
    agg_map = _normalize_map_sequence(agg_map)
    agg_metrics = _compute_similarity_metrics(agg_map, target_map)

    return {
        "family": family_name,
        "agg_map": agg_map.cpu(),
        "agg_metrics": agg_metrics,
        "scores_per_layer": score_vec.cpu(),
        "selected_indices": selected_idx.cpu(),
        "selected_layer_ids": layer_ids[selected_idx].cpu(),
        "selected_weights": weights.cpu(),
        "layer_reports": layer_reports,
    }


def _load_vggt4d_model(
    checkpoint: str,
    device: str,
):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    vggt_root = os.path.join(repo_root, "third_party", "VGGT4D")
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)

    from vggt4d.models.vggt4d import VGGTFor4D
    from vggt4d.utils.model_utils import organize_qk_dict

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"VGGT4D checkpoint not found: {checkpoint}")

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("Requested VGGT4D device cuda but CUDA is unavailable; falling back to cpu.")
        dev = torch.device("cpu")
    model = VGGTFor4D()
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(dev)
    return model, dev, organize_qk_dict


@torch.no_grad()
def _run_vggt4d_forward_qk(
    images: torch.Tensor,
    model,
    dev: torch.device,
    organize_qk_dict,
) -> dict[str, torch.Tensor]:
    imgs = images.to(dev)
    dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    with torch.no_grad():
        if dev.type == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                _, qk_dict, _, _ = model(imgs)
        else:
            _, qk_dict, _, _ = model(imgs)
    qk_org = organize_qk_dict(qk_dict, n_img=int(imgs.shape[0]))
    global_q = qk_org["global_tok_q"].detach().float().cpu()
    global_k = qk_org["global_tok_k"].detach().float().cpu()
    del qk_dict, qk_org
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return {"global_q": global_q, "global_k": global_k}


def save_layer_selection_heatmap(
    output_path: str,
    layer_ids: torch.Tensor,
    score_by_family: dict[str, torch.Tensor],
    selected_ids_by_family: dict[str, List[int]],
) -> None:
    family_names = list(score_by_family.keys())
    if not family_names:
        return
    mat = torch.stack([score_by_family[f].float() for f in family_names], dim=0).cpu().numpy()
    L = mat.shape[1]
    H_cell = 42
    W_cell = 44
    top_margin = 70
    left_margin = 130
    bottom_margin = 30
    right_margin = 20
    canvas_h = top_margin + mat.shape[0] * H_cell + bottom_margin
    canvas_w = left_margin + L * W_cell + right_margin
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    norm = np.clip((mat + 1.0) * 0.5, 0.0, 1.0)
    heat_u8 = (norm * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    heat_color = cv2.resize(heat_color, (L * W_cell, mat.shape[0] * H_cell), interpolation=cv2.INTER_NEAREST)
    canvas[top_margin:top_margin + mat.shape[0] * H_cell, left_margin:left_margin + L * W_cell] = heat_color

    cv2.putText(canvas, "LoGeR layer similarity to VGGT4D families", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "color range: score -1 -> +1", (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    for r, fam in enumerate(family_names):
        y = top_margin + r * H_cell + int(H_cell * 0.65)
        cv2.putText(canvas, fam, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)

    layer_ids_np = layer_ids.cpu().numpy().astype(np.int32)
    for c in range(L):
        x = left_margin + c * W_cell + 6
        cv2.putText(canvas, f"L{int(layer_ids_np[c])}", (x, top_margin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    for r, fam in enumerate(family_names):
        selected = set(selected_ids_by_family.get(fam, []))
        for c in range(L):
            lid = int(layer_ids_np[c])
            if lid in selected:
                x0 = left_margin + c * W_cell
                y0 = top_margin + r * H_cell
                x1 = x0 + W_cell - 1
                y1 = y0 + H_cell - 1
                cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 255), 2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
    print(f"Saved layer-selection heatmap to {output_path}")


def create_vggt4d_alignment_video(
    images: torch.Tensor,
    target_maps: dict[str, torch.Tensor],
    auto_maps: dict[str, torch.Tensor],
    output_path: str,
    fps: int,
) -> None:
    required = ["mean1", "var1", "mean2", "mean3", "var3", "dyn"]
    if any(k not in target_maps or k not in auto_maps for k in required):
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    T, _, H, W = images.shape
    gap = 4
    num_cols = 4
    num_rows = 3
    grid_w = W * num_cols + gap * (num_cols - 1)
    grid_h = H * num_rows + gap * (num_rows - 1)
    colorbar_w = 96 + gap

    def _up(seq_map: torch.Tensor) -> np.ndarray:
        up = F.interpolate(
            seq_map.unsqueeze(1).float().cpu(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        return up.numpy()

    prepared = [
        ("m1 target", _up(target_maps["mean1"])),
        ("m1 auto", _up(auto_maps["mean1"])),
        ("m2 target", _up(target_maps["mean2"])),
        ("m2 auto", _up(auto_maps["mean2"])),
        ("v1 target", _up(target_maps["var1"])),
        ("v1 auto", _up(auto_maps["var1"])),
        ("m3 target", _up(target_maps["mean3"])),
        ("m3 auto", _up(auto_maps["mean3"])),
        ("v3 target", _up(target_maps["var3"])),
        ("v3 auto", _up(auto_maps["var3"])),
        ("dyn target", _up(target_maps["dyn"])),
        ("dyn auto", _up(auto_maps["dyn"])),
    ]

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (grid_w + colorbar_w, grid_h),
    )
    for t in range(T):
        row_imgs = []
        for r in range(num_rows):
            row_panels = []
            for c in range(num_cols):
                idx = r * num_cols + c
                title, seq = prepared[idx]
                panel = _render_panel(
                    title,
                    _colorize_map(seq[t], cv2.COLORMAP_TURBO, value_range=(0.0, 1.0)),
                )
                row_panels.append(panel)
                if c < num_cols - 1:
                    row_panels.append(np.zeros((H, gap, 3), np.uint8))
            row_imgs.append(np.concatenate(row_panels, axis=1))
        frame = row_imgs[0]
        for row in row_imgs[1:]:
            frame = np.concatenate([frame, np.zeros((gap, grid_w, 3), np.uint8), row], axis=0)
        frame = _append_colorbar_legend(frame, cv2.COLORMAP_TURBO, title="Value")
        writer.write(frame)
    writer.release()
    print(f"Saved VGGT4D alignment video to {output_path}")


def run_vggt4d_loger_alignment(
    images: torch.Tensor,
    loger_q_layers: torch.Tensor,
    loger_k_layers: torch.Tensor,
    loger_layer_ids: torch.Tensor | None,
    world_points: torch.Tensor | None,
    local_points: torch.Tensor | None,
    camera_poses: torch.Tensor | None,
    vggt_ckpt: str,
    vggt_device: str,
    offsets_text: str,
    topk: int,
    max_frames: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, object]:
    """Run chunked VGGT4D reference extraction and auto-select LoGeR layers."""
    if loger_q_layers is None or loger_k_layers is None:
        raise ValueError("LoGeR per-layer raw global q/k stacks are required.")

    T_total = int(images.shape[0])
    use_frames = T_total if max_frames <= 0 else min(T_total, int(max_frames))
    if use_frames < 3:
        raise ValueError(f"Need at least 3 frames for alignment, got {use_frames}.")

    images_sub = images[:use_frames]
    loger_q_sub = loger_q_layers[:use_frames].float().cpu()
    loger_k_sub = loger_k_layers[:use_frames].float().cpu()
    world_points_sub = world_points[:use_frames] if world_points is not None else None
    local_points_sub = local_points[:use_frames] if local_points is not None else None
    camera_poses_sub = camera_poses[:use_frames] if camera_poses is not None else None
    if loger_layer_ids is None:
        L = loger_q_sub.shape[1]
        loger_layer_ids = torch.arange(1, 2 * L + 1, 2, dtype=torch.long)
    else:
        loger_layer_ids = loger_layer_ids.detach().cpu().long()

    offsets = _parse_offset_list(offsets_text)
    family_defs = [
        ("mean1", "qq_mean_layers"),
        ("var1", "qq_var_layers"),
        ("mean2", "qq_mean_layers"),
        ("mean3", "kk_mean_layers"),
        ("var3", "qk_var_layers"),
        ("kq_var_diag", "kq_var_layers"),
    ]
    target_keys = ["mean1", "var1", "mean2", "mean3", "var3", "kq_var_diag", "dyn"]
    candidate_keys = sorted({candidate_key for _, candidate_key in family_defs})

    margin = max((abs(int(o)) for o in offsets), default=0)
    effective_chunk_size = use_frames if chunk_size <= 0 else min(use_frames, int(chunk_size))
    if use_frames > effective_chunk_size and effective_chunk_size < 2 * margin + 1:
        raise ValueError(
            "VGGT4D chunk_size is too small for the requested offsets. "
            f"Need chunk_size >= {2 * margin + 1}, got {effective_chunk_size}."
        )

    if use_frames > effective_chunk_size:
        auto_overlap = 2 * margin
        effective_chunk_overlap = auto_overlap if chunk_overlap <= 0 else int(chunk_overlap)
        effective_chunk_overlap = max(effective_chunk_overlap, auto_overlap)
        effective_chunk_overlap = min(effective_chunk_overlap, effective_chunk_size - 1)
    else:
        effective_chunk_overlap = 0
    chunks = split_into_chunks(use_frames, effective_chunk_size, effective_chunk_overlap)

    print(
        "  VGGT4D chunk schedule: "
        f"{len(chunks)} chunk(s), size={effective_chunk_size}, "
        f"overlap={effective_chunk_overlap}, margin={margin}"
    )

    model, dev, organize_qk_dict = _load_vggt4d_model(vggt_ckpt, vggt_device)
    target_accum: dict[str, torch.Tensor] | None = None
    candidate_accum: dict[str, torch.Tensor] | None = None
    target_count = torch.zeros(use_frames, 1, 1, dtype=torch.float32)
    candidate_count = torch.zeros(use_frames, 1, 1, 1, dtype=torch.float32)
    patch_hw = (images_sub.shape[-2] // 14, images_sub.shape[-1] // 14)

    try:
        for ci, (start, end) in enumerate(chunks):
            chunk_images = images_sub[start:end]
            chunk_q = loger_q_sub[start:end]
            chunk_k = loger_k_sub[start:end]

            vggt_qk = _run_vggt4d_forward_qk(chunk_images, model, dev, organize_qk_dict)
            chunk_target_maps = _compute_vggt4d_family_targets(
                vggt_qk["global_q"], vggt_qk["global_k"], patch_hw, offsets,
            )
            chunk_candidates = _compute_loger_family_candidates_with_offsets(
                chunk_q, chunk_k, offsets,
            )

            local_frames = int(chunk_images.shape[0])
            keep_start = 0 if ci == 0 else min(margin, local_frames)
            keep_end = local_frames if ci == len(chunks) - 1 else max(keep_start, local_frames - margin)
            if keep_end <= keep_start:
                raise ValueError(
                    "VGGT4D chunk keep region is empty. "
                    f"chunk=({start}, {end}), keep=({keep_start}, {keep_end}), margin={margin}"
                )

            global_start = start + keep_start
            global_end = start + keep_end
            if target_accum is None:
                target_accum = {
                    key: torch.zeros(use_frames, *chunk_target_maps[key].shape[1:], dtype=torch.float32)
                    for key in target_keys
                }
                candidate_accum = {
                    key: torch.zeros(use_frames, *chunk_candidates[key].shape[1:], dtype=torch.float32)
                    for key in candidate_keys
                }

            target_count[global_start:global_end] += 1.0
            candidate_count[global_start:global_end] += 1.0
            for key in target_keys:
                target_accum[key][global_start:global_end] += chunk_target_maps[key][keep_start:keep_end]
            for key in candidate_keys:
                candidate_accum[key][global_start:global_end] += chunk_candidates[key][keep_start:keep_end]

            del vggt_qk, chunk_target_maps, chunk_candidates
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if target_accum is None or candidate_accum is None:
        raise ValueError("VGGT4D alignment produced no chunk outputs.")

    missing_target = torch.where(target_count[:, 0, 0] <= 0)[0]
    missing_candidate = torch.where(candidate_count[:, 0, 0, 0] <= 0)[0]
    if int(missing_target.numel()) > 0 or int(missing_candidate.numel()) > 0:
        raise ValueError(
            "VGGT4D chunk merge left uncovered frames. "
            f"missing_target={missing_target.tolist()} "
            f"missing_candidate={missing_candidate.tolist()}"
        )

    target_maps = {
        key: (target_accum[key] / target_count.clamp_min(1.0)).cpu()
        for key in target_keys
    }
    loger_candidates = {
        key: (candidate_accum[key] / candidate_count.clamp_min(1.0)).cpu()
        for key in candidate_keys
    }

    target_maps["dyn_raw"] = target_maps["dyn"].clone()
    target_dyn_post = _postprocess_dyn_map_like_vggt4d_demo(
        target_maps["dyn_raw"],
        images=images_sub,
        world_points=world_points_sub,
        local_points=local_points_sub,
        camera_poses=camera_poses_sub,
        cue_patch=None,
    )
    target_maps["dyn_coarse"] = target_dyn_post["coarse_dyn_patch"]
    target_maps["dyn_cluster"] = target_dyn_post["cluster_dyn_patch"]
    target_maps["dyn_mask"] = target_dyn_post["binary_mask_patch"]
    target_maps["dyn"] = target_dyn_post["refined_dyn_patch"]

    score_by_family: Dict[str, torch.Tensor] = {}
    selected_ids_by_family: Dict[str, List[int]] = {}
    auto_maps: Dict[str, torch.Tensor] = {}
    selection_report: Dict[str, object] = {}

    family_topk = {
        "mean1": max(1, int(topk)),
        "var1": max(1, min(int(topk), 3)),
        "mean2": max(1, int(topk)),
        "mean3": max(1, min(int(topk), 2)),
        "var3": max(1, min(int(topk), 2)),
        "kq_var_diag": max(1, min(int(topk), 2)),
    }

    for family, candidate_key in family_defs:
        sel = _auto_select_layers_for_family(
            family_name=family,
            candidate_layers=loger_candidates[candidate_key],
            target_map=target_maps[family],
            layer_ids=loger_layer_ids,
            topk=family_topk[family],
        )
        auto_maps[family] = sel["agg_map"]
        score_by_family[family] = sel["scores_per_layer"]
        selected_ids_by_family[family] = [int(x.item()) for x in sel["selected_layer_ids"]]
        selection_report[family] = {
            "candidate_type": candidate_key,
            "selected_layer_ids": selected_ids_by_family[family],
            "selected_weights": [float(x.item()) for x in sel["selected_weights"]],
            "agg_metrics": sel["agg_metrics"],
            "top_candidates": sorted(
                sel["layer_reports"],
                key=lambda d: d["metrics"]["score"],
                reverse=True,
            )[:10],
        }

    dyn_auto = _normalize_map_sequence(
        (1.0 - auto_maps["mean1"])
        * (1.0 - auto_maps["var1"])
        * auto_maps["mean2"]
        * (1.0 - auto_maps["mean3"])
        * auto_maps["var3"]
    ).cpu()
    auto_maps["dyn"] = dyn_auto
    dyn_metrics = _compute_similarity_metrics(dyn_auto, target_maps["dyn"])

    family_metrics = {
        fam: _compute_similarity_metrics(auto_maps[fam], target_maps[fam])
        for fam in ("mean1", "var1", "mean2", "mean3", "var3", "kq_var_diag")
    }

    return {
        "num_frames_used": use_frames,
        "offsets": offsets,
        "chunk_size": effective_chunk_size,
        "chunk_overlap": effective_chunk_overlap,
        "chunk_margin": margin,
        "num_chunks": len(chunks),
        "target_maps": target_maps,
        "target_dyn_postprocess": {
            "cluster_threshold": target_dyn_post["cluster_threshold"],
            "cluster_labels_patch": target_dyn_post["cluster_labels_patch"],
        },
        "auto_maps": auto_maps,
        "score_by_family": score_by_family,
        "selected_ids_by_family": selected_ids_by_family,
        "selection_report": selection_report,
        "family_metrics": family_metrics,
        "dyn_metrics": dyn_metrics,
        "loger_layer_ids": loger_layer_ids,
    }


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
    if args.run_vggt4d_alignment and not os.path.isfile(args.vggt4d_checkpoint):
        sys.exit(f"VGGT4D checkpoint not found: {args.vggt4d_checkpoint}")

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
        backbone.model.export_attn_debug = bool(args.debug_attention_vis)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        cue_kwargs = dict(
            k_intra=args.k_intra,
            use_attention_prior=not args.disable_attention_prior,
            support_time_decay=args.support_time_decay,
            support_temporal_weight=args.support_temporal_weight,
            support_affinity_weight=args.support_affinity_weight,
            support_static_weight=args.support_static_weight,
            sigma_pt=args.sigma_pt,
            tau_occ=args.tau_occ,
            alpha_1=args.alpha_1,
            alpha_3=args.alpha_3,
            attn_stat_fusion_weight=args.attn_stat_fusion_weight,
            attn_dyn_weight=args.attn_dyn_weight,
            attn_gate_power=args.attn_gate_power,
            attn_debias_kernel=args.attn_debias_kernel,
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
        all_frame_priors: List[torch.Tensor | None] = []
        all_attn_dynamic_patches: List[torch.Tensor | None] = []
        all_dyn4d_patches: List[torch.Tensor | None] = []
        all_dyn4d_qq_mean_patches: List[torch.Tensor | None] = []
        all_dyn4d_qk_var_patches: List[torch.Tensor | None] = []
        all_dyn4d_kk_mean_patches: List[torch.Tensor | None] = []
        all_local_points: List[torch.Tensor | None] = []
        all_world_points: List[torch.Tensor | None] = []
        all_camera_poses: List[torch.Tensor | None] = []
        all_global_q_raw_patchvecs: List[torch.Tensor | None] = []
        all_global_k_raw_patchvecs: List[torch.Tensor | None] = []
        all_global_q_raw_patchvec_layer_stacks: List[torch.Tensor | None] = []
        all_global_k_raw_patchvec_layer_stacks: List[torch.Tensor | None] = []
        merged_dyn4d_global_layer_ids: torch.Tensor | None = None
        all_frame_attn_cosine_shallow: List[torch.Tensor | None] = []
        all_frame_attn_cosine_deep: List[torch.Tensor | None] = []
        all_frame_attn_cosine_avg: List[torch.Tensor | None] = []
        all_frame_attn_key_cosine_l0: List[torch.Tensor | None] = []
        all_frame_attn_key_cosine_l4: List[torch.Tensor | None] = []
        all_frame_attn_key_cosine_shallow: List[torch.Tensor | None] = []
        all_frame_attn_key_cosine_deep: List[torch.Tensor | None] = []
        all_frame_attn_key_cosine_avg: List[torch.Tensor | None] = []
        all_frame_attn_cosine_query_layers: List[torch.Tensor | None] = []
        all_frame_attn_cosine_key_layers: List[torch.Tensor | None] = []
        merged_frame_attn_layer_ids: torch.Tensor | None = None

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
            all_frame_priors.append(geo.frame_attention_prior)
            all_attn_dynamic_patches.append(geo.attn_dynamic_patch)
            all_dyn4d_patches.append(geo.dyn4d_patch)
            all_dyn4d_qq_mean_patches.append(geo.dyn4d_qq_mean_patch)
            all_dyn4d_qk_var_patches.append(geo.dyn4d_qk_var_patch)
            all_dyn4d_kk_mean_patches.append(geo.dyn4d_kk_mean_patch)
            all_local_points.append(geo.local_points)
            all_world_points.append(geo.world_points)
            all_camera_poses.append(geo.camera_poses)
            all_global_q_raw_patchvecs.append(geo.global_q_raw_patchvec)
            all_global_k_raw_patchvecs.append(geo.global_k_raw_patchvec)
            all_global_q_raw_patchvec_layer_stacks.append(geo.global_q_raw_patchvec_layers)
            all_global_k_raw_patchvec_layer_stacks.append(geo.global_k_raw_patchvec_layers)
            if merged_dyn4d_global_layer_ids is None and geo.dyn4d_global_layer_ids is not None:
                merged_dyn4d_global_layer_ids = geo.dyn4d_global_layer_ids
            all_frame_attn_cosine_shallow.append(geo.frame_attn_cosine_shallow)
            all_frame_attn_cosine_deep.append(geo.frame_attn_cosine_deep)
            all_frame_attn_cosine_avg.append(geo.frame_attn_cosine_avg)
            all_frame_attn_key_cosine_l0.append(geo.frame_attn_key_cosine_l0)
            all_frame_attn_key_cosine_l4.append(geo.frame_attn_key_cosine_l4)
            all_frame_attn_key_cosine_shallow.append(geo.frame_attn_key_cosine_shallow)
            all_frame_attn_key_cosine_deep.append(geo.frame_attn_key_cosine_deep)
            all_frame_attn_key_cosine_avg.append(geo.frame_attn_key_cosine_avg)
            all_frame_attn_cosine_query_layers.append(geo.frame_attn_cosine_query_layers)
            all_frame_attn_cosine_key_layers.append(geo.frame_attn_cosine_key_layers)
            if merged_frame_attn_layer_ids is None and geo.frame_attn_cosine_layer_ids is not None:
                merged_frame_attn_layer_ids = geo.frame_attn_cosine_layer_ids

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
        merged_frame_attention_prior, merged_attn_dynamic_patch = merge_chunk_attention_priors(
            all_frame_priors,
            all_attn_dynamic_patches,
            args.chunk_overlap,
        )
        merged_dyn4d_patch = merge_chunk_sequence_tensor(
            all_dyn4d_patches, args.chunk_overlap,
        )
        merged_dyn4d_qq_mean_patch = merge_chunk_sequence_tensor(
            all_dyn4d_qq_mean_patches, args.chunk_overlap,
        )
        merged_dyn4d_qk_var_patch = merge_chunk_sequence_tensor(
            all_dyn4d_qk_var_patches, args.chunk_overlap,
        )
        merged_dyn4d_kk_mean_patch = merge_chunk_sequence_tensor(
            all_dyn4d_kk_mean_patches, args.chunk_overlap,
        )
        merged_local_points = merge_chunk_sequence_tensor(
            all_local_points, args.chunk_overlap,
        )
        merged_world_points = merge_chunk_sequence_tensor(
            all_world_points, args.chunk_overlap,
        )
        merged_camera_poses = merge_chunk_sequence_tensor(
            all_camera_poses, args.chunk_overlap,
        )
        merged_global_q_raw_patchvec = merge_chunk_sequence_tensor(
            all_global_q_raw_patchvecs, args.chunk_overlap,
        )
        merged_global_k_raw_patchvec = merge_chunk_sequence_tensor(
            all_global_k_raw_patchvecs, args.chunk_overlap,
        )
        merged_global_q_raw_patchvec_layers = merge_chunk_sequence_tensor(
            all_global_q_raw_patchvec_layer_stacks, args.chunk_overlap,
        )
        merged_global_k_raw_patchvec_layers = merge_chunk_sequence_tensor(
            all_global_k_raw_patchvec_layer_stacks, args.chunk_overlap,
        )
        merged_frame_attn_cosine_shallow = merge_chunk_sequence_tensor(
            all_frame_attn_cosine_shallow, args.chunk_overlap,
        )
        merged_frame_attn_cosine_deep = merge_chunk_sequence_tensor(
            all_frame_attn_cosine_deep, args.chunk_overlap,
        )
        merged_frame_attn_cosine_avg = merge_chunk_sequence_tensor(
            all_frame_attn_cosine_avg, args.chunk_overlap,
        )
        merged_frame_attn_key_cosine_l0 = merge_chunk_sequence_tensor(
            all_frame_attn_key_cosine_l0, args.chunk_overlap,
        )
        merged_frame_attn_key_cosine_l4 = merge_chunk_sequence_tensor(
            all_frame_attn_key_cosine_l4, args.chunk_overlap,
        )
        merged_frame_attn_key_cosine_shallow = merge_chunk_sequence_tensor(
            all_frame_attn_key_cosine_shallow, args.chunk_overlap,
        )
        merged_frame_attn_key_cosine_deep = merge_chunk_sequence_tensor(
            all_frame_attn_key_cosine_deep, args.chunk_overlap,
        )
        merged_frame_attn_key_cosine_avg = merge_chunk_sequence_tensor(
            all_frame_attn_key_cosine_avg, args.chunk_overlap,
        )
        merged_frame_attn_cosine_query_layers = merge_chunk_sequence_tensor(
            all_frame_attn_cosine_query_layers, args.chunk_overlap,
        )
        merged_frame_attn_cosine_key_layers = merge_chunk_sequence_tensor(
            all_frame_attn_cosine_key_layers, args.chunk_overlap,
        )
        global_q_raw_layer_maps = None
        global_k_raw_layer_maps = None
        if merged_global_q_raw_patchvec_layers is not None:
            global_q_raw_layer_maps = _summarize_patchvec_response_layers(
                merged_global_q_raw_patchvec_layers,
            )
        if merged_global_k_raw_patchvec_layers is not None:
            global_k_raw_layer_maps = _summarize_patchvec_response_layers(
                merged_global_k_raw_patchvec_layers,
            )
        global_dyn4d_layer_stats = None
        if (
            merged_global_q_raw_patchvec_layers is not None
            and merged_global_k_raw_patchvec_layers is not None
        ):
            global_dyn4d_layer_stats = _compute_dyn4d_token_gram_stats_per_layer(
                merged_global_q_raw_patchvec_layers,
                merged_global_k_raw_patchvec_layers,
                window_radius=2,
            )
        images_merged = torch.cat(merged_image_parts, dim=0) if merged_image_parts else images
        dyn4d_variant_maps = build_dyn4d_variant_maps(
            merged_global_q_raw_patchvec,
            merged_global_k_raw_patchvec,
            merged_global_q_raw_patchvec_layers,
            merged_global_k_raw_patchvec_layers,
            merged_dyn4d_global_layer_ids,
            images_merged,
            merged_world_points,
            merged_local_points,
            merged_camera_poses,
            cue.E_cue_patch,
        )
        manual_dyn_variant = next(
            (v for v in dyn4d_variant_maps if v.get("name") == "vggt4d_custom"),
            None,
        )
        if isinstance(manual_dyn_variant, dict) and isinstance(manual_dyn_variant.get("dyn4d_patch"), torch.Tensor):
            merged_dyn4d_patch = manual_dyn_variant["dyn4d_patch"]
        vggt_alignment_result: dict[str, object] | None = None
        if args.run_vggt4d_alignment:
            if (
                merged_global_q_raw_patchvec_layers is None
                or merged_global_k_raw_patchvec_layers is None
            ):
                print("Skip VGGT4D alignment: LoGeR per-layer raw global q/k stacks are unavailable.")
            else:
                print("\nRunning VGGT4D alignment for automatic LoGeR layer selection ...")
                t_align = time.time()
                vggt_alignment_result = run_vggt4d_loger_alignment(
                    images=images_merged,
                    loger_q_layers=merged_global_q_raw_patchvec_layers,
                    loger_k_layers=merged_global_k_raw_patchvec_layers,
                    loger_layer_ids=merged_dyn4d_global_layer_ids,
                    world_points=merged_world_points,
                    local_points=merged_local_points,
                    camera_poses=merged_camera_poses,
                    vggt_ckpt=args.vggt4d_checkpoint,
                    vggt_device=args.vggt4d_device,
                    offsets_text=args.vggt4d_offsets,
                    topk=args.vggt4d_select_topk,
                    max_frames=args.vggt4d_max_frames,
                    chunk_size=args.vggt4d_chunk_size,
                    chunk_overlap=args.vggt4d_chunk_overlap,
                )
                print(f"VGGT4D alignment done in {time.time() - t_align:.2f}s")
        print("\nMerged chunk outputs:")
        print_cue_output(cue)

        if merged_frame_attention_prior is not None:
            print(
                "Merged Stage-A frame attention prior: "
                f"shape={tuple(merged_frame_attention_prior.shape)}  "
                f"mean={merged_frame_attention_prior.mean().item():.4f}  "
                f"max={merged_frame_attention_prior.max().item():.4f}"
            )
        if merged_attn_dynamic_patch is not None:
            print(
                "Merged Stage-A attention feature: "
                f"shape={tuple(merged_attn_dynamic_patch.shape)}  "
                f"mean={merged_attn_dynamic_patch.mean().item():.4f}  "
                f"max={merged_attn_dynamic_patch.max().item():.4f}"
            )
        if merged_dyn4d_patch is not None:
            print(
                "Merged 4D_dyn feature: "
                f"shape={tuple(merged_dyn4d_patch.shape)}  "
                f"mean={merged_dyn4d_patch.mean().item():.4f}  "
                f"max={merged_dyn4d_patch.max().item():.4f}"
            )
        if merged_dyn4d_qq_mean_patch is not None:
            print(
                "Merged 4D qq_mean: "
                f"shape={tuple(merged_dyn4d_qq_mean_patch.shape)}  "
                f"mean={merged_dyn4d_qq_mean_patch.mean().item():.4f}  "
                f"max={merged_dyn4d_qq_mean_patch.max().item():.4f}"
            )
        if merged_dyn4d_qk_var_patch is not None:
            print(
                "Merged 4D qk_var: "
                f"shape={tuple(merged_dyn4d_qk_var_patch.shape)}  "
                f"mean={merged_dyn4d_qk_var_patch.mean().item():.4f}  "
                f"max={merged_dyn4d_qk_var_patch.max().item():.4f}"
            )
        if merged_dyn4d_kk_mean_patch is not None:
            print(
                "Merged 4D kk_mean: "
                f"shape={tuple(merged_dyn4d_kk_mean_patch.shape)}  "
                f"mean={merged_dyn4d_kk_mean_patch.mean().item():.4f}  "
                f"max={merged_dyn4d_kk_mean_patch.max().item():.4f}"
            )
        if merged_global_q_raw_patchvec is not None:
            print(
                "Merged raw global_q patchvec: "
                f"shape={tuple(merged_global_q_raw_patchvec.shape)}  "
                f"mean={merged_global_q_raw_patchvec.mean().item():.4f}  "
                f"max={merged_global_q_raw_patchvec.max().item():.4f}"
            )
        if merged_global_k_raw_patchvec is not None:
            print(
                "Merged raw global_k patchvec: "
                f"shape={tuple(merged_global_k_raw_patchvec.shape)}  "
                f"mean={merged_global_k_raw_patchvec.mean().item():.4f}  "
                f"max={merged_global_k_raw_patchvec.max().item():.4f}"
            )
        if merged_global_q_raw_patchvec_layers is not None:
            print(
                "Merged raw global_q layer-stack: "
                f"shape={tuple(merged_global_q_raw_patchvec_layers.shape)}"
            )
        if merged_global_k_raw_patchvec_layers is not None:
            print(
                "Merged raw global_k layer-stack: "
                f"shape={tuple(merged_global_k_raw_patchvec_layers.shape)}"
            )
        if merged_dyn4d_global_layer_ids is not None:
            print(f"4D_dyn global layer ids: {merged_dyn4d_global_layer_ids.tolist()}")
        if global_q_raw_layer_maps is not None:
            print(f"Per-layer global raw q maps: shape={tuple(global_q_raw_layer_maps.shape)}")
        if global_k_raw_layer_maps is not None:
            print(f"Per-layer global raw k maps: shape={tuple(global_k_raw_layer_maps.shape)}")
        if global_dyn4d_layer_stats is not None:
            for key, tensor in global_dyn4d_layer_stats.items():
                print(f"Per-layer global stat {key}: shape={tuple(tensor.shape)}")
        if dyn4d_variant_maps:
            print("Offline 4D_dyn comparison variants:")
            for variant in dyn4d_variant_maps:
                dyn_map = variant["dyn4d_patch"]
                if isinstance(dyn_map, torch.Tensor):
                    print(
                        f"  - {variant['title']}: "
                        f"radius={variant['radius']}  "
                        f"weights={variant['weights']}  "
                        f"mean={dyn_map.mean().item():.4f}  "
                        f"max={dyn_map.max().item():.4f}"
                    )
                    layer_map = variant.get("layer_map")
                    if layer_map is not None:
                        print(f"    layer_map={layer_map}")
        if vggt_alignment_result is not None:
            print("VGGT4D alignment summary:")
            print(
                f"  frames used={vggt_alignment_result['num_frames_used']}  "
                f"offsets={vggt_alignment_result['offsets']}"
            )
            print(
                f"  chunk_size={vggt_alignment_result['chunk_size']}  "
                f"chunk_overlap={vggt_alignment_result['chunk_overlap']}  "
                f"chunk_margin={vggt_alignment_result['chunk_margin']}  "
                f"num_chunks={vggt_alignment_result['num_chunks']}"
            )
            print(
                f"  dyn target postprocess: "
                f"cluster_threshold={vggt_alignment_result['target_dyn_postprocess']['cluster_threshold']:.4f}"
            )
            dyn_m = vggt_alignment_result["dyn_metrics"]
            print(
                f"  dyn metrics: score={dyn_m['score']:.4f}  "
                f"pearson={dyn_m['pearson']:.4f}  "
                f"iou20={dyn_m['iou20']:.4f}  "
                f"mae={dyn_m['mae']:.4f}"
            )
            fam_metrics = vggt_alignment_result["family_metrics"]
            for fam in ("mean1", "var1", "mean2", "mean3", "var3", "kq_var_diag"):
                fm = fam_metrics[fam]
                sel = vggt_alignment_result["selected_ids_by_family"][fam]
                print(
                    f"  {fam:10s} sel={sel}  score={fm['score']:.4f}  "
                    f"pearson={fm['pearson']:.4f}  iou20={fm['iou20']:.4f}"
                )
        if merged_frame_attn_cosine_shallow is not None:
            print(
                "Merged MUT3R-style frame cosine shallow: "
                f"shape={tuple(merged_frame_attn_cosine_shallow.shape)}  "
                f"mean={merged_frame_attn_cosine_shallow.mean().item():.4f}  "
                f"max={merged_frame_attn_cosine_shallow.max().item():.4f}"
            )
        if merged_frame_attn_cosine_deep is not None:
            print(
                "Merged MUT3R-style frame cosine deep: "
                f"shape={tuple(merged_frame_attn_cosine_deep.shape)}  "
                f"mean={merged_frame_attn_cosine_deep.mean().item():.4f}  "
                f"max={merged_frame_attn_cosine_deep.max().item():.4f}"
            )
        if merged_frame_attn_cosine_avg is not None:
            print(
                "Merged MUT3R-style query cosine avg: "
                f"shape={tuple(merged_frame_attn_cosine_avg.shape)}  "
                f"mean={merged_frame_attn_cosine_avg.mean().item():.4f}  "
                f"max={merged_frame_attn_cosine_avg.max().item():.4f}"
            )
        if merged_frame_attn_key_cosine_l0 is not None:
            print(
                "Merged Stage-A key cosine layer 0: "
                f"shape={tuple(merged_frame_attn_key_cosine_l0.shape)}  "
                f"mean={merged_frame_attn_key_cosine_l0.mean().item():.4f}  "
                f"max={merged_frame_attn_key_cosine_l0.max().item():.4f}"
            )
        if merged_frame_attn_key_cosine_l4 is not None:
            print(
                "Merged Stage-A key cosine layer 4: "
                f"shape={tuple(merged_frame_attn_key_cosine_l4.shape)}  "
                f"mean={merged_frame_attn_key_cosine_l4.mean().item():.4f}  "
                f"max={merged_frame_attn_key_cosine_l4.max().item():.4f}"
            )
        if merged_frame_attn_key_cosine_shallow is not None:
            print(
                "Merged MUT3R-style key cosine shallow: "
                f"shape={tuple(merged_frame_attn_key_cosine_shallow.shape)}  "
                f"mean={merged_frame_attn_key_cosine_shallow.mean().item():.4f}  "
                f"max={merged_frame_attn_key_cosine_shallow.max().item():.4f}"
            )
        if merged_frame_attn_key_cosine_deep is not None:
            print(
                "Merged MUT3R-style key cosine deep: "
                f"shape={tuple(merged_frame_attn_key_cosine_deep.shape)}  "
                f"mean={merged_frame_attn_key_cosine_deep.mean().item():.4f}  "
                f"max={merged_frame_attn_key_cosine_deep.max().item():.4f}"
            )
        if merged_frame_attn_key_cosine_avg is not None:
            print(
                "Merged MUT3R-style key cosine avg: "
                f"shape={tuple(merged_frame_attn_key_cosine_avg.shape)}  "
                f"mean={merged_frame_attn_key_cosine_avg.mean().item():.4f}  "
                f"max={merged_frame_attn_key_cosine_avg.max().item():.4f}"
            )
        if args.debug_attention_vis and merged_frame_attn_cosine_query_layers is not None:
            print(
                "Merged per-layer query cosine stack: "
                f"shape={tuple(merged_frame_attn_cosine_query_layers.shape)}"
            )
        if args.debug_attention_vis and merged_frame_attn_cosine_key_layers is not None:
            print(
                "Merged per-layer key cosine stack: "
                f"shape={tuple(merged_frame_attn_cosine_key_layers.shape)}"
            )
        if args.debug_attention_vis and merged_frame_attn_layer_ids is not None:
            print(f"Per-layer frame-attn ids: {merged_frame_attn_layer_ids.tolist()}")

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
                attn_dynamic_patch=merged_attn_dynamic_patch,
                dyn4d_patch=merged_dyn4d_patch,
                dyn4d_qq_mean_patch=merged_dyn4d_qq_mean_patch,
                dyn4d_qk_var_patch=merged_dyn4d_qk_var_patch,
                dyn4d_kk_mean_patch=merged_dyn4d_kk_mean_patch,
                global_q_raw_patchvec=merged_global_q_raw_patchvec,
                global_k_raw_patchvec=merged_global_k_raw_patchvec,
                frame_attention_prior=merged_frame_attention_prior,
                frame_attn_cosine_shallow=merged_frame_attn_cosine_shallow,
                frame_attn_cosine_deep=merged_frame_attn_cosine_deep,
                frame_attn_cosine_avg=merged_frame_attn_cosine_avg,
                frame_attn_key_cosine_shallow=merged_frame_attn_key_cosine_shallow,
                frame_attn_key_cosine_deep=merged_frame_attn_key_cosine_deep,
                frame_attn_key_cosine_avg=merged_frame_attn_key_cosine_avg,
            )
            if merged_frame_attention_prior is not None:
                attn_img_path = os.path.splitext(args.output_video)[0] + "_frame_attention.png"
                save_frame_attention_summary(merged_frame_attention_prior, attn_img_path)
            base, ext = os.path.splitext(args.output_video)
            if dyn4d_variant_maps:
                create_dyn4d_variant_video(
                    images=images_merged,
                    variant_maps=dyn4d_variant_maps,
                    output_path=base + "_4dyn_compare" + ext,
                    fps=args.fps,
                )
                vggt4d_direct_variant = next(
                    (v for v in dyn4d_variant_maps if v.get("name") == "vggt4d_direct"),
                    None,
                )
                if vggt4d_direct_variant is not None:
                    create_vggt4d_terms_video(
                        images=images_merged,
                        variant=vggt4d_direct_variant,
                        output_path=base + "_vggt4d_terms" + ext,
                        fps=args.fps,
                    )
                vggt4d_custom_variant = next(
                    (v for v in dyn4d_variant_maps if v.get("name") == "vggt4d_custom"),
                    None,
                )
                if vggt4d_custom_variant is not None:
                    create_vggt4d_custom_terms_video(
                        images=images_merged,
                        variant=vggt4d_custom_variant,
                        output_path=base + "_vggt4d_custom_terms" + ext,
                        fps=args.fps,
                    )
            if (
                args.debug_attention_vis
                and
                merged_frame_attn_cosine_query_layers is not None
                and merged_frame_attn_layer_ids is not None
            ):
                create_attention_layer_grid_video(
                    merged_frame_attn_cosine_query_layers,
                    merged_frame_attn_layer_ids,
                    base + "_query_layers" + ext,
                    args.fps,
                    prefix="Q",
                )
            if (
                args.debug_attention_vis
                and
                merged_frame_attn_cosine_key_layers is not None
                and merged_frame_attn_layer_ids is not None
            ):
                create_attention_layer_grid_video(
                    merged_frame_attn_cosine_key_layers,
                    merged_frame_attn_layer_ids,
                    base + "_key_layers" + ext,
                    args.fps,
                    prefix="K",
                )
            if (
                args.debug_attention_vis
                and global_q_raw_layer_maps is not None
                and merged_dyn4d_global_layer_ids is not None
            ):
                create_attention_layer_grid_video(
                    global_q_raw_layer_maps,
                    merged_dyn4d_global_layer_ids,
                    base + "_global_q_raw_layers" + ext,
                    args.fps,
                    prefix="GQ",
                )
            if (
                args.debug_attention_vis
                and global_k_raw_layer_maps is not None
                and merged_dyn4d_global_layer_ids is not None
            ):
                create_attention_layer_grid_video(
                    global_k_raw_layer_maps,
                    merged_dyn4d_global_layer_ids,
                    base + "_global_k_raw_layers" + ext,
                    args.fps,
                    prefix="GK",
                )
            if (
                args.debug_attention_vis
                and global_dyn4d_layer_stats is not None
                and merged_dyn4d_global_layer_ids is not None
            ):
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["qq_mean_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_qq_mean_layers" + ext,
                    args.fps,
                    prefix="QQmean",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["qq_mean_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_qq_mean_layers_contrast" + ext,
                    args.fps,
                    prefix="QQmean*",
                    vis_mode="contrast",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["qq_var_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_qq_var_layers" + ext,
                    args.fps,
                    prefix="QQvar",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["qq_var_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_qq_var_layers_contrast" + ext,
                    args.fps,
                    prefix="QQvar*",
                    vis_mode="contrast",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["kk_mean_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_kk_mean_layers" + ext,
                    args.fps,
                    prefix="KKmean",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["kk_mean_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_kk_mean_layers_contrast" + ext,
                    args.fps,
                    prefix="KKmean*",
                    vis_mode="contrast",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["qk_var_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_qk_var_layers" + ext,
                    args.fps,
                    prefix="QKvar",
                )
                create_attention_layer_grid_video(
                    global_dyn4d_layer_stats["qk_var_layers"],
                    merged_dyn4d_global_layer_ids,
                    base + "_global_qk_var_layers_contrast" + ext,
                    args.fps,
                    prefix="QKvar*",
                    vis_mode="contrast",
                )
            if vggt_alignment_result is not None:
                alignment_frames = int(vggt_alignment_result["num_frames_used"])
                create_vggt4d_alignment_video(
                    images=images_merged[:alignment_frames],
                    target_maps=vggt_alignment_result["target_maps"],
                    auto_maps=vggt_alignment_result["auto_maps"],
                    output_path=base + "_vggt4d_alignment" + ext,
                    fps=args.fps,
                )
                save_layer_selection_heatmap(
                    output_path=base + "_vggt4d_layer_selection.png",
                    layer_ids=vggt_alignment_result["loger_layer_ids"],
                    score_by_family=vggt_alignment_result["score_by_family"],
                    selected_ids_by_family=vggt_alignment_result["selected_ids_by_family"],
                )
                report_path = base + "_vggt4d_alignment_metrics.json"
                report = {
                    "num_frames_used": int(vggt_alignment_result["num_frames_used"]),
                    "offsets": vggt_alignment_result["offsets"],
                    "chunk_size": int(vggt_alignment_result["chunk_size"]),
                    "chunk_overlap": int(vggt_alignment_result["chunk_overlap"]),
                    "chunk_margin": int(vggt_alignment_result["chunk_margin"]),
                    "num_chunks": int(vggt_alignment_result["num_chunks"]),
                    "target_dyn_postprocess": {
                        "cluster_threshold": vggt_alignment_result["target_dyn_postprocess"]["cluster_threshold"],
                    },
                    "dyn_metrics": vggt_alignment_result["dyn_metrics"],
                    "family_metrics": vggt_alignment_result["family_metrics"],
                    "selected_ids_by_family": vggt_alignment_result["selected_ids_by_family"],
                    "selection_report": vggt_alignment_result["selection_report"],
                }
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                print(f"Saved VGGT4D alignment metrics to {report_path}")
        elif vggt_alignment_result is not None:
            if args.output:
                base = os.path.splitext(args.output)[0]
            else:
                base = os.path.join("results", "vggt4d_alignment_debug")
                os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
            save_layer_selection_heatmap(
                output_path=base + "_vggt4d_layer_selection.png",
                layer_ids=vggt_alignment_result["loger_layer_ids"],
                score_by_family=vggt_alignment_result["score_by_family"],
                selected_ids_by_family=vggt_alignment_result["selected_ids_by_family"],
            )
            report_path = base + "_vggt4d_alignment_metrics.json"
            report = {
                "num_frames_used": int(vggt_alignment_result["num_frames_used"]),
                "offsets": vggt_alignment_result["offsets"],
                "chunk_size": int(vggt_alignment_result["chunk_size"]),
                "chunk_overlap": int(vggt_alignment_result["chunk_overlap"]),
                "chunk_margin": int(vggt_alignment_result["chunk_margin"]),
                "num_chunks": int(vggt_alignment_result["num_chunks"]),
                "target_dyn_postprocess": {
                    "cluster_threshold": vggt_alignment_result["target_dyn_postprocess"]["cluster_threshold"],
                },
                "dyn_metrics": vggt_alignment_result["dyn_metrics"],
                "family_metrics": vggt_alignment_result["family_metrics"],
                "selected_ids_by_family": vggt_alignment_result["selected_ids_by_family"],
                "selection_report": vggt_alignment_result["selection_report"],
            }
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"Saved VGGT4D alignment metrics to {report_path}")

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
            if cue.C_dyn_explicit is not None:
                save_dict["C_dyn_explicit"] = cue.C_dyn_explicit
            if cue.C_dyn_implicit is not None:
                save_dict["C_dyn_implicit"] = cue.C_dyn_implicit
            if cue.C_dyn_fusion_max is not None:
                save_dict["C_dyn_fusion_max"] = cue.C_dyn_fusion_max
            if cue.C_dyn_fusion_soft_or is not None:
                save_dict["C_dyn_fusion_soft_or"] = cue.C_dyn_fusion_soft_or
            if cue.C_dyn_fusion_avg is not None:
                save_dict["C_dyn_fusion_avg"] = cue.C_dyn_fusion_avg
            if cue.C_dyn_fusion_addclip is not None:
                save_dict["C_dyn_fusion_addclip"] = cue.C_dyn_fusion_addclip
            if merged_frame_attention_prior is not None:
                save_dict["frame_attention_prior"] = merged_frame_attention_prior
            if merged_attn_dynamic_patch is not None:
                save_dict["attn_dynamic_patch"] = merged_attn_dynamic_patch
            if merged_dyn4d_patch is not None:
                save_dict["dyn4d_patch"] = merged_dyn4d_patch
            if merged_dyn4d_qq_mean_patch is not None:
                save_dict["dyn4d_qq_mean_patch"] = merged_dyn4d_qq_mean_patch
            if merged_dyn4d_qk_var_patch is not None:
                save_dict["dyn4d_qk_var_patch"] = merged_dyn4d_qk_var_patch
            if merged_dyn4d_kk_mean_patch is not None:
                save_dict["dyn4d_kk_mean_patch"] = merged_dyn4d_kk_mean_patch
            if merged_global_q_raw_patchvec is not None:
                save_dict["global_q_raw_patchvec"] = merged_global_q_raw_patchvec
            if merged_global_k_raw_patchvec is not None:
                save_dict["global_k_raw_patchvec"] = merged_global_k_raw_patchvec
            if merged_global_q_raw_patchvec_layers is not None:
                save_dict["global_q_raw_patchvec_layers"] = merged_global_q_raw_patchvec_layers
            if merged_global_k_raw_patchvec_layers is not None:
                save_dict["global_k_raw_patchvec_layers"] = merged_global_k_raw_patchvec_layers
            if merged_dyn4d_global_layer_ids is not None:
                save_dict["dyn4d_global_layer_ids"] = merged_dyn4d_global_layer_ids
            if global_q_raw_layer_maps is not None:
                save_dict["global_q_raw_layer_maps"] = global_q_raw_layer_maps
            if global_k_raw_layer_maps is not None:
                save_dict["global_k_raw_layer_maps"] = global_k_raw_layer_maps
            if global_dyn4d_layer_stats is not None:
                for key, tensor in global_dyn4d_layer_stats.items():
                    save_dict[key] = tensor
            if vggt_alignment_result is not None:
                save_dict["vggt4d_alignment_num_frames_used"] = int(vggt_alignment_result["num_frames_used"])
                save_dict["vggt4d_alignment_offsets"] = vggt_alignment_result["offsets"]
                save_dict["vggt4d_alignment_chunk_size"] = int(vggt_alignment_result["chunk_size"])
                save_dict["vggt4d_alignment_chunk_overlap"] = int(vggt_alignment_result["chunk_overlap"])
                save_dict["vggt4d_alignment_chunk_margin"] = int(vggt_alignment_result["chunk_margin"])
                save_dict["vggt4d_alignment_num_chunks"] = int(vggt_alignment_result["num_chunks"])
                save_dict["vggt4d_target_dyn_cluster_threshold"] = float(
                    vggt_alignment_result["target_dyn_postprocess"]["cluster_threshold"]
                )
                save_dict["vggt4d_alignment_dyn_metrics"] = vggt_alignment_result["dyn_metrics"]
                save_dict["vggt4d_alignment_family_metrics"] = vggt_alignment_result["family_metrics"]
                save_dict["vggt4d_alignment_selected_ids_by_family"] = vggt_alignment_result["selected_ids_by_family"]
                save_dict["vggt4d_alignment_selection_report"] = vggt_alignment_result["selection_report"]
                save_dict["vggt4d_alignment_layer_ids"] = vggt_alignment_result["loger_layer_ids"]
                cluster_labels = vggt_alignment_result["target_dyn_postprocess"].get("cluster_labels_patch")
                if isinstance(cluster_labels, torch.Tensor):
                    save_dict["vggt4d_target_dyn_cluster_labels_patch"] = cluster_labels
                for key, tensor in vggt_alignment_result["target_maps"].items():
                    if isinstance(tensor, torch.Tensor):
                        save_dict[f"vggt4d_target_{key}_patch"] = tensor
                for key, tensor in vggt_alignment_result["auto_maps"].items():
                    if isinstance(tensor, torch.Tensor):
                        save_dict[f"loger_auto_{key}_patch"] = tensor
                for key, tensor in vggt_alignment_result["score_by_family"].items():
                    if isinstance(tensor, torch.Tensor):
                        save_dict[f"vggt4d_alignment_score_{key}"] = tensor
            if dyn4d_variant_maps:
                dyn4d_variant_tensors = [
                    v["dyn4d_patch"] for v in dyn4d_variant_maps
                    if isinstance(v.get("dyn4d_patch"), torch.Tensor)
                ]
                if dyn4d_variant_tensors:
                    save_dict["dyn4d_variant_names"] = [v["name"] for v in dyn4d_variant_maps]
                    save_dict["dyn4d_variant_titles"] = [v["title"] for v in dyn4d_variant_maps]
                    save_dict["dyn4d_variant_radii"] = [v["radius"] for v in dyn4d_variant_maps]
                    save_dict["dyn4d_variant_weights"] = [v["weights"] for v in dyn4d_variant_maps]
                    save_dict["dyn4d_variant_patches"] = torch.stack(dyn4d_variant_tensors, dim=0)
                    vggt4d_direct_variant = next(
                        (v for v in dyn4d_variant_maps if v.get("name") == "vggt4d_direct"),
                        None,
                    )
                    if vggt4d_direct_variant is not None:
                        for key in ("mean1_patch", "var1_patch", "mean2_patch", "mean3_patch", "var3_patch"):
                            value = vggt4d_direct_variant.get(key)
                            if isinstance(value, torch.Tensor):
                                save_dict[f"vggt4d_direct_{key}"] = value
                        if "layer_map" in vggt4d_direct_variant:
                            save_dict["vggt4d_direct_layer_map"] = vggt4d_direct_variant["layer_map"]
                    vggt4d_custom_variant = next(
                        (v for v in dyn4d_variant_maps if v.get("name") == "vggt4d_custom"),
                        None,
                    )
                    if vggt4d_custom_variant is not None:
                        for key in (
                            "shallow_kk_mean_patch",
                            "shallow_qk_var_patch",
                            "shallow_term_patch",
                            "middle_qq_mean_patch",
                            "middle_term_patch",
                            "deep_qq_var_patch",
                            "deep_term_patch",
                            "coarse_dyn_patch",
                            "cluster_dyn_patch",
                            "binary_mask_patch",
                        ):
                            value = vggt4d_custom_variant.get(key)
                            if isinstance(value, torch.Tensor):
                                save_dict[f"vggt4d_custom_{key}"] = value
                        if "cluster_threshold" in vggt4d_custom_variant:
                            save_dict["vggt4d_custom_cluster_threshold"] = vggt4d_custom_variant["cluster_threshold"]
                        value = vggt4d_custom_variant.get("cluster_labels_patch")
                        if isinstance(value, torch.Tensor):
                            save_dict["vggt4d_custom_cluster_labels_patch"] = value
                        if "layer_map" in vggt4d_custom_variant:
                            save_dict["vggt4d_custom_layer_map"] = vggt4d_custom_variant["layer_map"]
                    loger_lite_variant = next(
                        (v for v in dyn4d_variant_maps if v.get("name") == "loger_lite_m3v3"),
                        None,
                    )
                    if loger_lite_variant is not None:
                        for key in ("mean3_patch", "var3_patch"):
                            value = loger_lite_variant.get(key)
                            if isinstance(value, torch.Tensor):
                                save_dict[f"loger_lite_{key}"] = value
                        if "layer_map" in loger_lite_variant:
                            save_dict["loger_lite_layer_map"] = loger_lite_variant["layer_map"]
            if merged_frame_attn_cosine_shallow is not None:
                save_dict["frame_attn_cosine_shallow"] = merged_frame_attn_cosine_shallow
            if merged_frame_attn_cosine_deep is not None:
                save_dict["frame_attn_cosine_deep"] = merged_frame_attn_cosine_deep
            if merged_frame_attn_cosine_avg is not None:
                save_dict["frame_attn_cosine_avg"] = merged_frame_attn_cosine_avg
            if merged_frame_attn_key_cosine_l0 is not None:
                save_dict["frame_attn_key_cosine_l0"] = merged_frame_attn_key_cosine_l0
            if merged_frame_attn_key_cosine_l4 is not None:
                save_dict["frame_attn_key_cosine_l4"] = merged_frame_attn_key_cosine_l4
            if merged_frame_attn_key_cosine_shallow is not None:
                save_dict["frame_attn_key_cosine_shallow"] = merged_frame_attn_key_cosine_shallow
            if merged_frame_attn_key_cosine_deep is not None:
                save_dict["frame_attn_key_cosine_deep"] = merged_frame_attn_key_cosine_deep
            if merged_frame_attn_key_cosine_avg is not None:
                save_dict["frame_attn_key_cosine_avg"] = merged_frame_attn_key_cosine_avg
            if merged_frame_attn_cosine_query_layers is not None:
                save_dict["frame_attn_cosine_query_layers"] = merged_frame_attn_cosine_query_layers
            if merged_frame_attn_cosine_key_layers is not None:
                save_dict["frame_attn_cosine_key_layers"] = merged_frame_attn_cosine_key_layers
            if merged_frame_attn_layer_ids is not None:
                save_dict["frame_attn_cosine_layer_ids"] = merged_frame_attn_layer_ids
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
