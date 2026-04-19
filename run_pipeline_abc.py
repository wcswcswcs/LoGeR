#!/usr/bin/env python3
"""
Full five-stage Semantic Prior Pipeline  (A → B → C → D → E)

Processes a video in **chunks**: for each chunk the pipeline runs all
five stages sequentially, and the TTT Write Controller (Stage E) commits
the new fast weights W_{m+1} that the next chunk's Geometry Backbone
(Stage A) will read.

Stages:
  A  LoGeR Geometry Backbone  → GeometryOutput + WriteCacheOutput
  B  Dynamic Cue Extractor    → CueOutput
  C  Video Masklet Front-end  → MaskletOutput
  D  Semantic Prior Generator  → PriorOutput (A_tok)
  E  TTT Write Controller      → WriteResult (W_{m+1})

Usage::

    CUDA_VISIBLE_DEVICES=0 conda run -n loger python run_pipeline_abc.py \\
        --input data/examples/office \\
        --config ckpts/LoGeR/original_config.yaml \\
        --checkpoint ckpts/LoGeR/latest.pt \\
        --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \\
        --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
        --detector gdino \\
        --gdino_config Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \\
        --gdino_checkpoint /path/to/groundingdino_swint_ogc.pth \\
        --chunk_size 32 \\
        --output_video results/office_full_pipeline.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

GSAM2_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grounded-SAM-2")
if GSAM2_ROOT not in sys.path:
    sys.path.insert(0, GSAM2_ROOT)

from loger.pipeline.geometry_backbone import (
    GeometryOutput,
    WriteCacheOutput,
    LoGeRGeometryBackbone,
    load_images as loger_load_images,
)
from loger.pipeline.dynamic_cue_extractor import CueOutput, DynamicCueExtractor
from loger.pipeline.video_masklet_frontend import (
    MaskletOutput,
    VideoMaskletFrontend,
    SEMANTIC_GROUP_NAMES,
)
from loger.pipeline.semantic_prior_generator import (
    PriorOutput,
    SemanticPriorGenerator,
)
from loger.pipeline.ttt_write_controller import (
    WriteResult,
    TTTWriteController,
)

from run_geometry_backbone_inference import (
    collect_image_paths as collect_image_paths_geo,
    print_geometry_output,
)
from inference_dynamic_cue_extractor import print_cue_output


# ---------------------------------------------------------------------------
# Per-masklet C_dyn scoring (retained from v2 for visualisation)
# ---------------------------------------------------------------------------
def score_masklets_by_cdyn(
    mo: MaskletOutput, cue: CueOutput,
) -> Dict[int, float]:
    """Average C_dyn inside each masklet's visible mask region."""
    C_dyn = cue.C_dyn
    T_cue, H_p, W_p = C_dyn.shape
    H, W = mo.frame_height, mo.frame_width

    cdyn_full = F.interpolate(
        C_dyn.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(1)

    T = min(mo.num_frames, T_cue)
    scores: Dict[int, float] = {}
    for j in range(mo.num_masklets):
        total, n = 0.0, 0
        for t in range(T):
            if not mo.V_mask[j, t]:
                continue
            mask_t = mo.M_mask[j, t].bool()
            cnt = mask_t.sum().item()
            if cnt == 0:
                continue
            total += cdyn_full[t][mask_t].sum().item()
            n += cnt
        scores[j] = total / n if n > 0 else 0.0
    return scores


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------
_to_tensor = transforms.ToTensor()


def load_images_tensor(paths: list) -> torch.Tensor:
    return torch.stack([_to_tensor(Image.open(p).convert("RGB")) for p in paths])


# ---------------------------------------------------------------------------
# Chunk splitter
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full 5-stage Semantic Prior Pipeline (A→B→C→D→E) with chunk loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--input", required=True, help="Image folder or video file.")
    p.add_argument("--output_video", default="results/pipeline_full.mp4")
    p.add_argument("--output_pt", default=None, help="Save outputs as .pt file.")
    p.add_argument("--save_frames", default=None)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)

    # -- Chunk scheduling ---------------------------------------------------
    p.add_argument("--chunk_size", type=int, default=0,
                    help="Frames per chunk (0 = all frames as one chunk).")
    p.add_argument("--chunk_overlap", type=int, default=0)

    # -- Stage A: LoGeR -----------------------------------------------------
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap_size", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=0)
    p.add_argument("--se3", action="store_true", default=None)

    # -- Stage B: Dynamic Cue Extractor ------------------------------------
    p.add_argument("--k_intra", type=int, default=3)
    p.add_argument("--sigma_pt", type=float, default=0.25)
    p.add_argument("--tau_occ", type=float, default=0.05)

    # -- Stage C: Video Masklet Front-end ----------------------------------
    p.add_argument("--sam2_checkpoint", required=True)
    p.add_argument("--sam2_model_cfg", required=True)
    p.add_argument("--detector", choices=["gdino", "yoloe"], default="gdino")
    p.add_argument("--gdino_config", default=None)
    p.add_argument("--gdino_checkpoint", default=None)
    p.add_argument("--yoloe_model", default="yoloe-11l-seg.pt")
    p.add_argument("--ann_frame_idx", type=int, default=0)
    p.add_argument("--max_thing_objects", type=int, default=15)
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--thing_prompts", default=None)
    p.add_argument("--stuff_prompts", default=None)

    # -- Stage E: TTT Write Controller -------------------------------------
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=1.0)

    # -- Video output -------------------------------------------------------
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--mask_alpha", type=float, default=0.40)

    return p


# ---------------------------------------------------------------------------
# Visualisation
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


def _render_prior_panel(
    rgb_np: np.ndarray,
    prior_map: Optional[np.ndarray],
    t: int,
) -> np.ndarray:
    """Left panel: A_pix write-allow heatmap."""
    if prior_map is not None:
        heat = cv2.applyColorMap(
            (prior_map * 255).clip(0, 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )
        heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        panel = np.ascontiguousarray(
            (rgb_np.astype(np.float32) * 0.35
             + heat_rgb.astype(np.float32) * 0.65).astype(np.uint8)
        )
    else:
        panel = rgb_np.copy()

    cv2.putText(panel, f"A_pix (write prior)  frame {t}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"A_pix (write prior)  frame {t}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


def _render_mask_panel(
    rgb_np: np.ndarray,
    mo: MaskletOutput,
    t: int,
    alpha: float,
    dyn_scores: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Right panel: tracked masks with per-masklet dynamism score."""
    overlay = rgb_np.copy()
    for j in range(mo.num_masklets):
        if not mo.V_mask[j, t]:
            continue
        mask = mo.M_mask[j, t].numpy().astype(bool)
        if mask.sum() == 0:
            continue
        colour = np.array(_PALETTE[j % len(_PALETTE)], dtype=np.uint8)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1 - alpha)
            + colour.astype(np.float32) * alpha
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, tuple(int(c) for c in colour), 2)

        box = mo.B_mask[j, t].numpy().astype(int)
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tuple(int(c) for c in colour), 2)

        label = mo.L_sem[j] if j < len(mo.L_sem) else "?"
        ds = dyn_scores.get(j, 0.0) if dyn_scores else 0.0
        text = f"{label} dyn={ds:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(overlay, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                      tuple(int(c) for c in colour), -1)
        cv2.putText(overlay, text, (x1 + 2, max(th + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    info = f"Masklets  frame {t}  total: {mo.num_masklets}"
    cv2.putText(overlay, info, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, info, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def create_video(
    images: torch.Tensor,
    mo: MaskletOutput,
    output_path: str,
    fps: int,
    alpha: float,
    prior: Optional[PriorOutput] = None,
    per_masklet_dyn: Optional[Dict[int, float]] = None,
    save_frames_dir: Optional[str] = None,
) -> None:
    """Side-by-side video: [A_pix write prior | Masklet + dyn score]."""
    T = images.shape[0]
    H, W = mo.frame_height, mo.frame_width
    gap = 4

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2 + gap, H),
    )
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    prior_maps = None
    if prior is not None:
        A_pix = prior.A_pix  # [T_cue, H_p, W_p]
        prior_maps = F.interpolate(
            A_pix.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(1).numpy()

    for t in range(T):
        rgb_np = (images[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if rgb_np.shape[0] != H or rgb_np.shape[1] != W:
            rgb_np = cv2.resize(rgb_np, (W, H))

        pmap = prior_maps[t] if prior_maps is not None and t < prior_maps.shape[0] else None
        left = _render_prior_panel(rgb_np, pmap, t)
        right = _render_mask_panel(rgb_np, mo, t, alpha, dyn_scores=per_masklet_dyn)

        separator = np.zeros((H, gap, 3), dtype=np.uint8)
        combined = np.concatenate([left, separator, right], axis=1)

        frame_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
        if save_frames_dir:
            cv2.imwrite(os.path.join(save_frames_dir, f"frame_{t:05d}.jpg"), frame_bgr)

    writer.release()
    print(f"Saved video to {output_path}  ({T} frames, {fps} FPS)")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def print_masklet_output(mo: MaskletOutput) -> None:
    print(f"\n{'='*72}")
    print("MaskletOutput summary  (Stage C)")
    print(f"{'='*72}")
    print(f"  num_masklets : {mo.num_masklets}")
    print(f"  num_frames   : {mo.num_frames}")
    print(f"  frame_size   : {mo.frame_height} x {mo.frame_width}")
    jt = mo.debug.get("J_thing", "?")
    js = mo.debug.get("J_stuff", "?")
    print(f"  thing/stuff  : {jt} / {js}")
    if mo.num_masklets > 0:
        print()
        hdr = (f"  {'ID':>3s}  {'Type':14s}  {'Label':22s}  {'Group':22s}  "
               f"{'Birth':>5s}  {'Vis':>7s}")
        print(hdr)
        print("  " + "-" * len(hdr))
        for j in range(mo.num_masklets):
            vis = mo.V_mask[j].sum().item()
            g = mo.G_sem[j].item()
            gn = SEMANTIC_GROUP_NAMES.get(g, "?")
            lbl = mo.L_sem[j] if j < len(mo.L_sem) else "?"
            bf = mo.birth_frame[j] if j < len(mo.birth_frame) else -1
            stype = mo.source_type[j] if j < len(mo.source_type) else "?"
            print(f"  {j:3d}  {stype:14s}  {lbl:22s}  {gn:22s}  "
                  f"{bf:5d}  {vis:5.0f}/{mo.num_frames}")
    print(f"{'='*72}\n")


def print_prior_output(prior: PriorOutput) -> None:
    print(f"\n{'='*72}")
    print("PriorOutput summary  (Stage D)")
    print(f"{'='*72}")
    print(f"  A_mask shape   : {tuple(prior.A_mask.shape)}")
    print(f"  A_pix shape    : {tuple(prior.A_pix.shape)}")
    print(f"  A_tok shape    : {tuple(prior.A_tok.shape)}")
    print(f"  A_special      : {prior.A_special:.3f}")
    print(f"  A_tok  mean    : {prior.A_tok.mean().item():.4f}")
    print(f"  A_tok  min     : {prior.A_tok.min().item():.4f}")
    print(f"  A_tok  max     : {prior.A_tok.max().item():.4f}")
    rho = prior.debug.get("rho_suppr_chunk", 0)
    print(f"  rho_suppr      : {rho:.4f}")
    print(f"{'='*72}\n")


def print_write_result(wr: WriteResult) -> None:
    print(f"\n{'='*72}")
    print("WriteResult summary  (Stage E)")
    print(f"{'='*72}")
    n = len(wr.w0)
    print(f"  TTT layers committed: {n}")
    for li in range(n):
        d = wr.debug.get(f"layer_{li}", {})
        lam = d.get("lambda_write", "?")
        mp = d.get("mean_prior", "?")
        print(f"    layer {li}: lambda={lam}, mean_prior={mp}")
    print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    if args.config and not os.path.isfile(args.config):
        sys.exit(f"Config not found: {args.config}")
    if not os.path.isfile(args.checkpoint):
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # ===================================================================
    # Collect all images
    # ===================================================================
    image_paths, temp_dir = collect_image_paths_geo(
        args.input, args.start_frame, args.end_frame, args.stride,
    )
    if not image_paths:
        sys.exit("No images found.")
    total_frames = len(image_paths)
    print(f"Collected {total_frames} images.")

    # Full-resolution images for Stage C
    images_full = load_images_tensor(image_paths)
    _, _, H_full, W_full = images_full.shape
    print(f"Full-res images: {tuple(images_full.shape)}")

    # LoGeR-resolution images for Stage A
    target_w, target_h = args.resolution if args.resolution else (None, None)
    images_loger = loger_load_images(image_paths, target_w=target_w, target_h=target_h)
    print(f"LoGeR-res images: {tuple(images_loger.shape)}")

    # ===================================================================
    # Chunk schedule
    # ===================================================================
    chunks = split_into_chunks(total_frames, args.chunk_size, args.chunk_overlap)
    print(f"\nChunk schedule: {len(chunks)} chunk(s), "
          f"size={args.chunk_size or total_frames}, overlap={args.chunk_overlap}")
    for ci, (s, e) in enumerate(chunks):
        print(f"  chunk {ci}: frames [{s}, {e})")

    # ===================================================================
    # Build models (persistent across chunks)
    # ===================================================================
    # Stage A
    backbone_kwargs: dict = dict(
        device=args.device,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
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

    # Stage B
    extractor = DynamicCueExtractor(
        k_intra=args.k_intra,
        sigma_pt=args.sigma_pt,
        tau_occ=args.tau_occ,
    )

    # Stage C
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

    # Stage D
    prior_gen = SemanticPriorGenerator()

    # Stage E
    controller = TTTWriteController(
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        device=args.device,
    )

    # ===================================================================
    # Chunk-by-chunk processing loop
    # ===================================================================
    ttt_state: Optional[Dict] = None
    all_geo: List[GeometryOutput] = []
    all_cue: List[CueOutput] = []
    all_masklet: List[MaskletOutput] = []
    all_prior: List[PriorOutput] = []

    for ci, (start, end) in enumerate(chunks):
        print(f"\n{'#'*72}")
        print(f"# Chunk {ci}/{len(chunks)-1}  frames [{start}, {end})")
        print(f"{'#'*72}")

        chunk_loger = images_loger[start:end]
        chunk_full = images_full[start:end]

        # ---- Stage A ----
        print(f"\n  Stage A: Geometry Backbone ...")
        t0 = time.time()
        geo, write_cache = backbone.run(
            chunk_loger,
            ttt_state=ttt_state,
            cache_ttt_primitives=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    done in {time.time() - t0:.2f}s")
        print_geometry_output(geo)
        all_geo.append(geo)

        # ---- Stage B ----
        print(f"  Stage B: Dynamic Cue Extractor ...")
        t0 = time.time()
        cue = extractor.run(geo)
        print(f"    done in {time.time() - t0:.2f}s")
        print_cue_output(cue)
        all_cue.append(cue)

        # ---- Stage C ----
        print(f"  Stage C: Video Masklet Front-end ...")
        t0 = time.time()
        mo = _run_stage_c_lazy(build_kwargs, frontend_kwargs, chunk_full, ci)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    done in {time.time() - t0:.2f}s")
        print_masklet_output(mo)
        all_masklet.append(mo)

        # ---- Stage D ----
        print(f"  Stage D: Semantic Prior Generator ...")
        t0 = time.time()
        prior = prior_gen.run(cue, mo, geo)
        print(f"    done in {time.time() - t0:.2f}s")
        print_prior_output(prior)
        all_prior.append(prior)

        # ---- Stage E ----
        if write_cache.num_ttt_layers > 0:
            print(f"  Stage E: TTT Write Controller ...")
            t0 = time.time()
            wr = controller.run(write_cache, prior.A_tok, device=args.device)
            print(f"    done in {time.time() - t0:.2f}s")
            print_write_result(wr)

            ttt_state = {"w0": wr.w0, "w1": wr.w1, "w2": wr.w2}
        else:
            print("  Stage E: skipped (no TTT layers cached)")

        del geo, cue, write_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===================================================================
    # Visualisation: use first chunk results for now (single-chunk mode)
    # ===================================================================
    if all_masklet and all_cue:
        mo = all_masklet[0]
        cue = all_cue[0]
        prior = all_prior[0] if all_prior else None
        dyn_scores = score_masklets_by_cdyn(mo, cue)

        print("\nPer-masklet C_dyn scores:")
        print(f"  {'ID':>3s}  {'Label':20s}  {'Type':14s}  {'C_dyn':>6s}")
        print("  " + "-" * 50)
        for j in range(mo.num_masklets):
            lbl = mo.L_sem[j] if j < len(mo.L_sem) else "?"
            stype = mo.source_type[j] if j < len(mo.source_type) else "?"
            ds = dyn_scores.get(j, 0.0)
            print(f"  {j:3d}  {lbl:20s}  {stype:14s}  {ds:6.3f}")

        create_video(
            images_full[:chunks[0][1]],
            mo, args.output_video,
            fps=args.fps, alpha=args.mask_alpha,
            prior=prior, per_masklet_dyn=dyn_scores,
            save_frames_dir=args.save_frames,
        )

    if args.output_pt and all_masklet:
        os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
        mo = all_masklet[0]
        save_dict = {
            "M_mask": mo.M_mask,
            "V_mask": mo.V_mask,
            "L_sem": mo.L_sem,
            "G_sem": mo.G_sem,
            "W_sem": mo.W_sem,
        }
        if all_prior:
            save_dict["A_tok"] = all_prior[0].A_tok
            save_dict["A_pix"] = all_prior[0].A_pix
        torch.save(save_dict, args.output_pt)
        print(f"Saved output to {args.output_pt}")

    import shutil
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# Lazy Stage C builder (so SAM2 + detector are only loaded when needed)
# ---------------------------------------------------------------------------
_frontend_instance: Optional[VideoMaskletFrontend] = None


def _run_stage_c_lazy(
    build_kwargs: dict, frontend_kwargs: dict,
    chunk_images: torch.Tensor, chunk_idx: int,
) -> MaskletOutput:
    global _frontend_instance
    if _frontend_instance is None:
        print(f"    Building VideoMaskletFrontend (detector={build_kwargs.get('detector_type','?')}) ...")
        _frontend_instance = VideoMaskletFrontend.from_config(
            **build_kwargs, **frontend_kwargs,
        )
    return _frontend_instance.run(chunk_images)


if __name__ == "__main__":
    main()
