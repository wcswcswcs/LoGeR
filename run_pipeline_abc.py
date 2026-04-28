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
        --sam_backend sam31_multiplex \\
        --sam31_checkpoint ckpts/SAM3/sam3.1_multiplex.pt \\
        --sam2_checkpoint /home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt \\
        --sam2_model_cfg configs/sam2.1/sam2.1_hiera_l.yaml \\
        --detector yoloe \\
        --yoloe_model yoloe-11l-seg.pt \\
        --chunk_size 32 \\
        --output_video results/office_full_pipeline.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

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

from loger.pipeline.geometry_backbone import (
    GeometryOutput,
    WriteCacheOutput,
    LoGeRGeometryBackbone,
    load_images as loger_load_images,
)
from loger.pipeline.dynamic_cue_extractor import CueOutput, DynamicCueExtractor
from loger.pipeline.video_masklet_frontend import (
    VideoMaskletFrontend,
    MaskletOutput,
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
from loger.utils.rotation import mat_to_quat, quat_to_mat

from run_geometry_backbone_inference import (
    collect_image_paths as collect_image_paths_geo,
    print_geometry_output,
)
from inference_dynamic_cue_extractor import print_cue_output


DEFAULT_SAM3_CHECKPOINT = "ckpts/SAM3/sam3.pt"
DEFAULT_SAM31_CHECKPOINT = "ckpts/SAM3/sam3.1_multiplex.pt"
DEFAULT_SAM2_CHECKPOINT = "/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt"
DEFAULT_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_VIDEO_THING_PROMPTS = (
    "person,people,singer,dancer,performer,guitar,microphone,"
    "microphone stand,door,desk,table,window,monitor,chair"
)
DEFAULT_VIDEO_STUFF_PROMPTS = "floor,wall,ceiling"


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


def merge_chunk_tensor_tail_trim(
    tensors: List[torch.Tensor], overlap: int,
) -> Optional[torch.Tensor]:
    """Merge external chunks by trimming the tail-overlap of every non-final
    chunk. This is the closest match to the original long-context LoGeR
    scheduling when ``chunk_overlap`` is aligned with ``overlap_size``."""
    if not tensors:
        return None

    merged_parts: List[torch.Tensor] = []
    num_chunks = len(tensors)
    for i, tensor in enumerate(tensors):
        trim = overlap if i < num_chunks - 1 else 0
        if trim > 0:
            if tensor.shape[0] <= trim:
                continue
            merged_parts.append(tensor[:-trim])
        else:
            merged_parts.append(tensor)

    if not merged_parts:
        return None
    return torch.cat(merged_parts, dim=0)


def _rebuild_batched_raw_window(
    geo: GeometryOutput,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """Rebuild a batched raw-prediction dict from ``GeometryOutput.raw_predictions``.

    ``Pi3`` merge utilities expect tensors with a leading batch dimension.
    ``GeometryOutput.raw_predictions`` stores the same tensors already moved to
    CPU and squeezed along batch.  This helper restores ``[B=1, ...]`` layout
    and injects the global window range so we can reuse the original merge code.
    """
    raw = {}
    keep_keys = {
        "points",
        "local_points",
        "conf",
        "camera_poses",
        "local_camera_poses",
        "camera_qvec",
        "local_camera_qvec",
        "metric",
        "frame_attention_prior",
        "attn_dynamic_patch",
        "dyn4d_patch",
        "dyn4d_qq_mean_patch",
        "dyn4d_qk_var_patch",
        "dyn4d_kk_mean_patch",
        "global_q_raw_patchvec",
        "global_k_raw_patchvec",
        "global_q_raw_patchvec_layers",
        "global_k_raw_patchvec_layers",
        "frame_attn_cosine_shallow",
        "frame_attn_cosine_deep",
        "frame_attn_cosine_avg",
        "frame_attn_key_cosine_l0",
        "frame_attn_key_cosine_l4",
        "frame_attn_key_cosine_shallow",
        "frame_attn_key_cosine_deep",
        "frame_attn_key_cosine_avg",
        "frame_attn_cosine_query_layers",
        "frame_attn_cosine_key_layers",
    }

    for key, value in geo.raw_predictions.items():
        if key not in keep_keys or value is None or not torch.is_tensor(value):
            continue
        raw[key] = value.unsqueeze(0)

    raw["_window_start"] = int(start)
    raw["_window_end"] = int(end)
    return raw


def _merge_external_window_predictions(
    backbone: LoGeRGeometryBackbone,
    windows_raw: List[Dict[str, Any]],
    *,
    window_size: int,
    overlap_size: int,
    reset_every: int,
    se3: bool,
) -> Dict[str, Any]:
    """Merge externally scheduled windows using the original Pi3 helpers."""
    model = backbone.model
    model._last_window_size = window_size
    model._last_overlap_size = overlap_size

    align_on_resets_without_explicit_pose = reset_every > 0 and not se3
    if se3 or align_on_resets_without_explicit_pose:
        return model._merge_windowed_predictions_sim3(
            windows_raw,
            allow_scale=False,
            reset_every=reset_every,
            reuse_transform_within_reset_block=align_on_resets_without_explicit_pose,
        )
    return model._merge_windowed_predictions(
        windows_raw,
        window_size,
        overlap_size,
    )


def run_geometry_eval_external_exact(
    *,
    backbone: LoGeRGeometryBackbone,
    controller: TTTWriteController,
    images_loger: torch.Tensor,
    chunks: List[Tuple[int, int]],
    args: argparse.Namespace,
) -> GeometryOutput:
    """Exact external window orchestrator for geometry-only parity checks.

    Each external chunk is treated as one LoGeR window.  We keep the original
    model math but move the sequence scheduling / reset logic out of
    ``Pi3.forward`` and reuse Pi3's own merge utilities at the end.
    """
    if args.chunk_size <= 0:
        raise ValueError("external exact geometry orchestrator requires chunk_size > 0")
    if args.chunk_size != args.window_size or args.chunk_overlap != args.overlap_size:
        raise ValueError(
            "For exact parity, external chunking must match internal windowing: "
            "chunk_size == window_size and chunk_overlap == overlap_size"
        )

    ttt_state: Optional[Dict[str, Any]] = None
    window_raw_predictions: List[Dict[str, Any]] = []

    for ci, (start, end) in enumerate(chunks):
        print(f"\n{'#'*72}")
        print(f"# Exact Geometry Window {ci}/{len(chunks)-1}  frames [{start}, {end})")
        print(f"{'#'*72}")

        if args.reset_every > 0 and ci > 0 and ci % args.reset_every == 0 and ttt_state is not None:
            preserved_history = ttt_state.get("history")
            ttt_state = {
                "w0": [None] * len(ttt_state.get("w0", [])),
                "w1": [None] * len(ttt_state.get("w1", [])),
                "w2": [None] * len(ttt_state.get("w2", [])),
            }
            if preserved_history is not None:
                ttt_state["history"] = preserved_history
            print(f"  External reset_every triggered at window {ci}: cleared TTT fast weights, preserved SWA history")

        chunk_loger = images_loger[start:end]

        print("  Stage A: Geometry Backbone ...")
        t0 = time.time()
        use_controller_write_through = (
            args.ttt_write_mode == "native" and args.native_write_through_controller
        )
        stage_a_result = backbone.run(
            chunk_loger,
            ttt_state=ttt_state,
            cache_ttt_primitives=False,
            window_size=args.window_size,
            overlap_size=args.overlap_size,
            reset_every=0,
            se3=False,
        )
        geo = stage_a_result
        write_cache = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    done in {time.time() - t0:.2f}s")
        print_geometry_output(geo)

        window_raw_predictions.append(_rebuild_batched_raw_window(geo, start, end))

        if use_controller_write_through:
            print("  Stage E: TTT Write Controller (native write-through) ...")
            t0 = time.time()
            native_state = backbone.get_ttt_state()
            write_cache = _build_native_write_cache_from_state(
                native_state,
                num_frames=geo.num_frames,
                patch_grid=geo.patch_grid,
            )
            wr = controller.run(
                write_cache,
                A_tok=None,
                B_chunk_geo=None,
                device=args.device,
            )
            print(f"    done in {time.time() - t0:.2f}s")
            print_write_result(wr)

            ttt_state = {"w0": wr.w0, "w1": wr.w1, "w2": wr.w2}
            if wr.history is not None:
                ttt_state["history"] = wr.history
        else:
            ttt_state = backbone.get_ttt_state()
            print("  Stage E: skipped (native write-through from Stage A state)")

        del geo, write_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged_raw = _merge_external_window_predictions(
        backbone,
        window_raw_predictions,
        window_size=args.window_size,
        overlap_size=args.overlap_size,
        reset_every=args.reset_every,
        se3=bool(args.se3),
    )

    T = images_loger.shape[0]
    H = images_loger.shape[2]
    W = images_loger.shape[3]
    patch_h, patch_w = H // 14, W // 14
    return backbone._postprocess(
        merged_raw,
        images_loger.unsqueeze(0),
        T,
        H,
        W,
        patch_h,
        patch_w,
    )


def _clone_ttt_tensor_list(xs: List[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
    return [x.detach().cpu().clone() if x is not None else None for x in xs]


def _clone_ttt_history(
    history: Optional[List[Optional[Dict[str, torch.Tensor]]]],
) -> Optional[List[Optional[Dict[str, torch.Tensor]]]]:
    if history is None:
        return None
    cloned: List[Optional[Dict[str, torch.Tensor]]] = []
    for entry in history:
        if entry is None:
            cloned.append(None)
        else:
            cloned.append({
                "k": entry["k"].detach().cpu().clone(),
                "v": entry["v"].detach().cpu().clone(),
            })
    return cloned


def _build_native_write_cache_from_state(
    ttt_state: Optional[Dict[str, Any]],
    *,
    num_frames: int,
    patch_grid: Tuple[int, int],
) -> WriteCacheOutput:
    """Build a lightweight native WriteCache from provisional fast weights.

    Native write-through does not need replay primitives (q/k/v/lr).  Stage A
    has already computed the updated fast weights, so Stage E only needs a
    small cache carrying those provisional weights through TTTWriteController.
    """
    if ttt_state is None:
        raise RuntimeError("native TTT write-through requested but Stage A produced no TTT state")

    w0 = _clone_ttt_tensor_list(ttt_state.get("w0", []))
    w1 = _clone_ttt_tensor_list(ttt_state.get("w1", []))
    w2 = _clone_ttt_tensor_list(ttt_state.get("w2", []))
    if not (len(w0) == len(w1) == len(w2)):
        raise RuntimeError("invalid native TTT state: w0/w1/w2 layer counts differ")

    return WriteCacheOutput(
        layer_caches=[],
        w0_provisional=w0,
        w1_provisional=w1,
        w2_provisional=w2,
        history_provisional=_clone_ttt_history(ttt_state.get("history")),
        num_frames=num_frames,
        patch_grid=patch_grid,
        num_ttt_layers=len(w0),
    )


def build_timestamps_for_output(image_paths: List[str], input_path: str) -> List[float]:
    """Best-effort timestamp loader, matching demo_viser.py behavior."""
    input_rgb_dir = Path(input_path)
    if input_rgb_dir.is_file():
        return [float(i) for i in range(len(image_paths))]

    rgb_txt_path = input_rgb_dir.parent / "rgb.txt"
    if rgb_txt_path.exists():
        name_to_ts: Dict[str, float] = {}
        with open(rgb_txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                name_to_ts[Path(parts[1]).name] = ts
        ts_list = [name_to_ts.get(Path(p).name, None) for p in image_paths]
        if all(t is not None for t in ts_list):
            return [float(t) for t in ts_list]

    timestamps_txt = input_rgb_dir / "timestamps.txt"
    if timestamps_txt.exists():
        with open(timestamps_txt, "r") as f:
            raw_lines = [l.strip() for l in f.readlines()
                         if l.strip() and not l.strip().startswith("#")]
        ts_list: List[float] = []
        for i in range(min(len(raw_lines), len(image_paths))):
            try:
                ts_list.append(float(raw_lines[i]))
            except ValueError:
                ts_list.append(float(i))
        for i in range(len(ts_list), len(image_paths)):
            ts_list.append(float(i))
        return ts_list

    return [float(i) for i in range(len(image_paths))]


def write_trajectory_txt(
    output_path: Path,
    timestamps: List[float],
    translations: List[List[float]],
    quaternions: List[List[float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, t, q in zip(timestamps, translations, quaternions):
            f.write(
                f"{ts:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
            )


def _average_pose_transforms(transforms: torch.Tensor) -> torch.Tensor:
    """Average SE(3) transforms from a small set of candidates."""
    if transforms.shape[0] == 1:
        return transforms[0]

    rotations = transforms[:, :3, :3]
    translations = transforms[:, :3, 3]
    quats = mat_to_quat(rotations)
    ref = quats[0:1]
    sign = torch.sign((quats * ref).sum(dim=-1, keepdim=True))
    sign[sign == 0] = 1
    quats = quats * sign
    quat_mean = quats.mean(dim=0)
    quat_mean = quat_mean / quat_mean.norm().clamp_min(1e-8)

    out = torch.eye(4, dtype=transforms.dtype)
    out[:3, :3] = quat_to_mat(quat_mean)
    out[:3, 3] = translations.mean(dim=0)
    return out


def align_chunk_geometry_outputs(
    chunks: List[GeometryOutput], overlap: int,
) -> List[GeometryOutput]:
    """Align each later chunk to the previous chunk using shared overlap poses."""
    if len(chunks) <= 1 or overlap <= 0:
        return chunks

    aligned: List[GeometryOutput] = [chunks[0]]
    for curr in chunks[1:]:
        prev = aligned[-1]
        ov = min(overlap, prev.camera_poses.shape[0], curr.camera_poses.shape[0])
        if ov <= 0:
            aligned.append(curr)
            continue

        prev_poses = prev.camera_poses[-ov:]
        curr_poses = curr.camera_poses[:ov]
        correction_candidates = torch.matmul(prev_poses, torch.linalg.inv(curr_poses))
        correction = _average_pose_transforms(correction_candidates)

        new_cam = torch.matmul(correction.unsqueeze(0), curr.camera_poses)

        world = curr.world_points
        flat = world.reshape(-1, 3)
        ones = torch.ones(flat.shape[0], 1, dtype=flat.dtype)
        homog = torch.cat([flat, ones], dim=-1)
        transformed = (homog @ correction.T)[..., :3].reshape_as(world)

        aligned.append(GeometryOutput(
            local_points=curr.local_points,
            world_points=transformed,
            camera_poses=new_cam,
            confidence=curr.confidence,
            patch_meta=curr.patch_meta,
            token_type=curr.token_type,
            frame_attention_prior=curr.frame_attention_prior,
            attn_dynamic_patch=curr.attn_dynamic_patch,
            dyn4d_patch=curr.dyn4d_patch,
            dyn4d_qq_mean_patch=curr.dyn4d_qq_mean_patch,
            dyn4d_qk_var_patch=curr.dyn4d_qk_var_patch,
            dyn4d_kk_mean_patch=curr.dyn4d_kk_mean_patch,
            global_q_raw_patchvec=curr.global_q_raw_patchvec,
            global_k_raw_patchvec=curr.global_k_raw_patchvec,
            global_q_raw_patchvec_layers=curr.global_q_raw_patchvec_layers,
            global_k_raw_patchvec_layers=curr.global_k_raw_patchvec_layers,
            dyn4d_global_layer_ids=curr.dyn4d_global_layer_ids,
            frame_attn_cosine_shallow=curr.frame_attn_cosine_shallow,
            frame_attn_cosine_deep=curr.frame_attn_cosine_deep,
            frame_attn_cosine_avg=curr.frame_attn_cosine_avg,
            frame_attn_key_cosine_l0=curr.frame_attn_key_cosine_l0,
            frame_attn_key_cosine_l4=curr.frame_attn_key_cosine_l4,
            frame_attn_key_cosine_shallow=curr.frame_attn_key_cosine_shallow,
            frame_attn_key_cosine_deep=curr.frame_attn_key_cosine_deep,
            frame_attn_key_cosine_avg=curr.frame_attn_key_cosine_avg,
            frame_attn_cosine_query_layers=curr.frame_attn_cosine_query_layers,
            frame_attn_cosine_key_layers=curr.frame_attn_cosine_key_layers,
            frame_attn_cosine_layer_ids=curr.frame_attn_cosine_layer_ids,
            num_frames=curr.num_frames,
            pointmap_resolution=curr.pointmap_resolution,
            patch_grid=curr.patch_grid,
            raw_predictions=curr.raw_predictions,
        ))
    return aligned


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
    p.add_argument("--output_txt", default=None, help="Save merged camera trajectory in TUM format.")
    p.add_argument("--save_frames", default=None)
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--geometry_eval_mode", action="store_true",
                   help="Geometry-only evaluation mode: skip Stages B/C/D and export merged Stage-A trajectory/results.")

    # -- Chunk scheduling ---------------------------------------------------
    p.add_argument("--chunk_size", type=int, default=0,
                    help="Frames per chunk (0 = all frames as one chunk).")
    p.add_argument("--chunk_overlap", type=int, default=2)

    # -- Stage A: LoGeR -----------------------------------------------------
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--overlap_size", type=int, default=3)
    p.add_argument("--reset_every", type=int, default=0)
    p.add_argument("--se3", action="store_true", default=None)
    p.add_argument("--geometry_edge_rtol", type=float, default=0.03,
                   help="Depth-edge confidence suppression used by Stage A. Set to 0.0 for demo_viser parity.")
    p.add_argument("--stage_memory_swap", type=int, default=1,
                   help="Move inactive Stage-A/Stage-C models to CPU between stages to keep pipeline GPU memory low (1/0).")

    # -- Stage B: Dynamic Cue Extractor ------------------------------------
    p.add_argument("--k_intra", type=int, default=3,
                   help="Temporal window-width parameter for Stage B; support candidates come from [t-k_intra//2, t+k_intra//2], and up to 4 views are sampled uniformly.")
    p.add_argument("--sigma_pt", type=float, default=0.25)
    p.add_argument("--tau_occ", type=float, default=0.05)

    # -- Stage C: Video Masklet Front-end -----------------------------------
    p.add_argument("--sam_backend", choices=["sam2", "sam3", "sam31_multiplex"], default="sam31_multiplex",
                   help="Stage-C SAM backend. The quality path uses sam31_multiplex.")
    p.add_argument("--tracker_backend", default="sam2",
                   choices=["sam2", "edgetam", "cutie", "efficientsam3", "efficient"],
                   help="Compatibility argument from the old efficient frontend; ignored by VideoMaskletFrontend.")
    p.add_argument("--sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    p.add_argument("--sam2_model_cfg", default=DEFAULT_SAM2_MODEL_CFG)
    p.add_argument("--edgetam_checkpoint", default=None,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--edgetam_model_cfg", default=None,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--sam3_checkpoint", default=DEFAULT_SAM3_CHECKPOINT,
                   help="SAM3 checkpoint used when --sam_backend=sam3.")
    p.add_argument("--sam31_checkpoint", default=DEFAULT_SAM31_CHECKPOINT,
                   help="SAM3.1 multiplex checkpoint used when --sam_backend=sam31_multiplex.")
    p.add_argument("--efficientsam3_checkpoint", default=None,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_repo_id", default=None,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_filename", default=None,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_cache_dir", default="ckpts/EfficientSAM3",
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_backbone_type", default="efficientvit",
                   choices=["repvit", "tinyvit", "efficientvit"],
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_model_name", default="b0",
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_text_encoder_type", default=None,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--efficientsam3_text_context_length", type=int, default=77,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--cutie_max_internal_size", type=int, default=480,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--sam3_cutie_sam_confidence_threshold", type=float, default=0.10,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--sam3_cutie_detection_frame_count", type=int, default=6,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--sam3_cutie_max_prompt_dets_per_label", type=int, default=4,
                   help="Compatibility argument from the old efficient frontend; ignored.")
    p.add_argument("--sam3_cutie_use_yoloe", type=int, default=1,
                   help="Compatibility argument from the old efficient frontend; ignored.")
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
    p.add_argument("--sam31_text_track_labels", default="person",
                   help="Comma-separated labels that use SAM3.1 text-prompt tracking; use 'all' for every detected label.")
    p.add_argument("--sam31_direct_text_prompt_labels", default="person",
                   help="Comma-separated THING labels directly queried with SAM3.1 text prompts independent of detector hits.")
    p.add_argument("--sam31_direct_text_prompt_frame_count", type=int, default=1,
                   help="Number of direct SAM3.1 text-prompt frames per chunk.")
    p.add_argument("--sam31_structure_prompt_labels", default="wall,floor,ceiling",
                   help="Comma-separated structure labels directly queried by Stage C.")
    p.add_argument("--sam31_structure_prompt_frame_count", type=int, default=1,
                   help="Number of discovery frames per chunk used for direct SAM3.1 structure prompt detection.")
    p.add_argument("--sam31_structure_prompt_chunk_stride", type=int, default=1,
                   help="Stage C runs direct SAM3.1 structure prompts every N chunks; 1 = every chunk, 0 = disabled.")
    p.add_argument("--sam31_person_refresh_prompt_frames", type=int, default=1,
                   help="Stage C SAM3.1 extra person/people text refresh prompt frames per chunk.")
    p.add_argument("--sam31_nontext_object_prompt_budget", type=int, default=0,
                   help="Stage C max non-text YOLOE candidates propagated with SAM3.1 object prompts per chunk; 0 keeps the default path fast.")
    p.add_argument("--sam31_nontext_object_prompt_min_support", type=int, default=2,
                   help="Stage C minimum detector support for non-seed non-text object prompts.")
    p.add_argument("--sam31_text_object_prompt_budget", type=int, default=0,
                   help="Max unmatched YOLOE candidates for text-tracked labels to force through SAM3.1 object prompts per label/chunk.")
    p.add_argument("--sam31_nontext_sparse_support", type=int, default=1,
                   help="Stage C keeps non-text YOLOE thing masks as sparse support without SAM3.1 propagation (1/0).")
    p.add_argument("--sam31_max_text_prompt_objects", type=int, default=0,
                   help="Stage C caps text-prompt objects propagated per SAM3.1 query; 0 keeps all.")
    p.add_argument("--sam31_max_internal_objects", type=int, default=16,
                   help="SAM3.1 multiplex runtime object slots per query.")
    p.add_argument("--sam31_max_movable_objects", type=int, default=8,
                   help="Max movable thing prompt tracks per chunk.")
    p.add_argument("--sam31_max_static_objects", type=int, default=2,
                   help="Max static thing prompt tracks per chunk.")
    p.add_argument("--sam31_max_structure_objects", type=int, default=3,
                   help="Max tracked structure prompt objects per chunk.")
    p.add_argument("--detector", choices=["gdino", "yoloe"], default="yoloe",
                   help="Detector used by Video Masklet Front-end.")
    p.add_argument("--gdino_config", default=None)
    p.add_argument("--gdino_checkpoint", default=None)
    p.add_argument("--yoloe_model", default="yoloe-11l-seg.pt")
    p.add_argument("--yoloe_batch_size", type=int, default=4)
    p.add_argument("--yoloe_imgsz", type=int, default=0)
    p.add_argument("--ann_frame_idx", type=int, default=0)
    p.add_argument("--discovery_frame_stride", type=int, default=4,
                   help="Run semantic discovery every N frames inside each chunk.")
    p.add_argument("--max_thing_objects", type=int, default=24)
    p.add_argument("--box_threshold", type=float, default=0.30)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--thing_prompts", default=DEFAULT_VIDEO_THING_PROMPTS)
    p.add_argument("--stuff_prompts", default=DEFAULT_VIDEO_STUFF_PROMPTS)

    # -- Stage E: TTT Write Controller -------------------------------------
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=1.0)
    p.add_argument("--ttt_write_mode", choices=["semantic", "unity_replay", "native"], default="semantic",
                   help="Stage E mode: semantic=use Stage D prior, unity_replay=all-one prior replay, native=use model provisional W directly.")
    p.add_argument("--native_write_through_controller", action="store_true",
                   help="In exact external geometry mode, route native fast-weight state through TTTWriteController write-through.")

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
    print(f"  Elig_pix shape : {tuple(prior.Elig_pix.shape)}")
    print(f"  r_mask shape   : {tuple(prior.r_mask.shape)}")
    print(f"  A_pix shape    : {tuple(prior.A_pix.shape)}")
    print(f"  A_tok shape    : {tuple(prior.A_tok.shape)}")
    print(f"  A_special      : {prior.A_special:.3f}")
    print(f"  B_chunk_geo    : {prior.B_chunk_geo:.4f}")
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
    mode = wr.debug.get("mode", "?")
    print(f"  mode               : {mode}")
    print(f"  TTT layers committed: {n}")
    for li in range(n):
        d = wr.debug.get(f"layer_{li}", {})
        lam = d.get("lambda_write", "?")
        mp = d.get("mean_prior", "?")
        bg = d.get("budget_geo", "?")
        print(f"    layer {li}: lambda={lam}, mean_prior={mp}, budget_geo={bg}")
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
    if args.geometry_eval_mode and args.ttt_write_mode == "semantic":
        sys.exit("geometry_eval_mode cannot be used with ttt_write_mode=semantic; use native or unity_replay.")
    if not args.geometry_eval_mode:
        if args.sam_backend == "sam2":
            if not args.sam2_checkpoint or not os.path.isfile(args.sam2_checkpoint):
                sys.exit(f"SAM2 checkpoint not found: {args.sam2_checkpoint}")
            if not args.sam2_model_cfg:
                sys.exit("sam2_model_cfg is required when --sam_backend=sam2.")
        elif args.sam_backend == "sam3":
            if not args.sam3_checkpoint or not os.path.isfile(args.sam3_checkpoint):
                sys.exit(f"SAM3 checkpoint not found: {args.sam3_checkpoint}")
        elif args.sam_backend == "sam31_multiplex":
            if args.sam31_checkpoint and not os.path.isfile(args.sam31_checkpoint):
                sys.exit(f"SAM3.1 checkpoint not found: {args.sam31_checkpoint}")
    if args.geometry_eval_mode and args.ttt_write_mode == "native":
        if args.chunk_size > 0 and args.chunk_size != args.window_size:
            print(f"[warn] native LoGeR parity is best when chunk_size ({args.chunk_size}) == window_size ({args.window_size}).")
        if args.chunk_overlap != args.overlap_size:
            print(f"[warn] native LoGeR parity is best when chunk_overlap ({args.chunk_overlap}) == overlap_size ({args.overlap_size}).")

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

    images_full: Optional[torch.Tensor]
    if args.geometry_eval_mode:
        images_full = None
        H_full = W_full = 0
        print("Full-res images: skipped (--geometry_eval_mode)")
    else:
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
        edge_rtol=args.geometry_edge_rtol,
        update_ttt_weights=(args.ttt_write_mode == "native"),
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
    extractor: Optional[DynamicCueExtractor]
    if args.geometry_eval_mode:
        extractor = None
    else:
        extractor = DynamicCueExtractor(
            k_intra=args.k_intra,
            sigma_pt=args.sigma_pt,
            tau_occ=args.tau_occ,
        )

    # Stage C
    frontend_kwargs: dict = {}
    build_kwargs: dict = {}
    if not args.geometry_eval_mode:
        frontend_kwargs = dict(
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            ann_frame_idx=args.ann_frame_idx,
            discovery_frame_stride=args.discovery_frame_stride,
            max_thing_objects=args.max_thing_objects,
        )
        if args.thing_prompts:
            frontend_kwargs["thing_prompts"] = [s.strip() for s in args.thing_prompts.split(",")]
        if args.stuff_prompts:
            frontend_kwargs["stuff_prompts"] = [s.strip() for s in args.stuff_prompts.split(",")]

        build_kwargs = dict(
            sam_backend=args.sam_backend,
            sam3_checkpoint=args.sam3_checkpoint,
            sam31_checkpoint=args.sam31_checkpoint,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_model_cfg=args.sam2_model_cfg,
            detector_type=args.detector,
            gdino_config=args.gdino_config,
            gdino_checkpoint=args.gdino_checkpoint,
            yoloe_model=args.yoloe_model,
            yoloe_batch_size=args.yoloe_batch_size,
            yoloe_imgsz=args.yoloe_imgsz,
            sam31_offload_video_to_cpu=args.sam31_offload_video_to_cpu,
            sam31_offload_outputs_to_cpu=args.sam31_offload_outputs_to_cpu,
            sam31_offload_sam_during_detection=args.sam31_offload_sam_during_detection,
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
            sam31_max_movable_objects=args.sam31_max_movable_objects,
            sam31_max_static_objects=args.sam31_max_static_objects,
            sam31_max_structure_objects=args.sam31_max_structure_objects,
            device=args.device,
        )

    need_semantic_prior = (args.ttt_write_mode == "semantic") and (not args.geometry_eval_mode)
    prior_gen = SemanticPriorGenerator() if need_semantic_prior else None

    # Stage E
    controller = TTTWriteController(
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        device=args.device,
        write_mode=args.ttt_write_mode,
    )

    if args.geometry_eval_mode and args.chunk_size > 0:
        print("\nUsing exact external geometry orchestrator.")
        merged_geo = run_geometry_eval_external_exact(
            backbone=backbone,
            controller=controller,
            images_loger=images_loger,
            chunks=chunks,
            args=args,
        )

        merged_camera_poses = merged_geo.camera_poses
        merged_local_points = merged_geo.local_points
        merged_world_points = merged_geo.world_points
        merged_confidence = merged_geo.confidence

        if args.output_txt and merged_camera_poses is not None:
            timestamps = build_timestamps_for_output(image_paths, args.input)
            twc = merged_camera_poses[:, :3, 3]
            qwc = mat_to_quat(merged_camera_poses[:, :3, :3])
            S = min(len(timestamps), twc.shape[0], qwc.shape[0])
            write_trajectory_txt(
                Path(args.output_txt),
                timestamps[:S],
                twc[:S].tolist(),
                qwc[:S].tolist(),
            )
            print(f"Saved trajectory to {args.output_txt}")

        if args.output_pt:
            os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
            save_dict: Dict[str, torch.Tensor] = {}
            if merged_local_points is not None:
                save_dict["local_points"] = merged_local_points
            if merged_world_points is not None:
                save_dict["points"] = merged_world_points
            if merged_camera_poses is not None:
                save_dict["camera_poses"] = merged_camera_poses
            if merged_confidence is not None:
                save_dict["conf"] = merged_confidence
            torch.save(save_dict, args.output_pt)
            print(f"Saved output to {args.output_pt}")
        return

    # ===================================================================
    # Chunk-by-chunk processing loop
    # ===================================================================
    ttt_state: Optional[Dict] = None
    all_geo: List[GeometryOutput] = []
    all_cue: List[Optional[CueOutput]] = []
    all_masklet: List[Optional[MaskletOutput]] = []
    all_prior: List[Optional[PriorOutput]] = []

    for ci, (start, end) in enumerate(chunks):
        print(f"\n{'#'*72}")
        print(f"# Chunk {ci}/{len(chunks)-1}  frames [{start}, {end})")
        print(f"{'#'*72}")

        if args.stage_memory_swap and str(args.device).startswith("cuda"):
            _offload_stage_c_frontend_to_cpu()
            _ensure_backbone_on_device(backbone, args.device)

        if args.reset_every > 0 and ci > 0 and ci % args.reset_every == 0 and ttt_state is not None:
            preserved_history = ttt_state.get("history")
            ttt_state = {
                "w0": [None] * len(ttt_state.get("w0", [])),
                "w1": [None] * len(ttt_state.get("w1", [])),
                "w2": [None] * len(ttt_state.get("w2", [])),
            }
            if preserved_history is not None:
                ttt_state["history"] = preserved_history
            print(f"  External reset_every triggered at chunk {ci}: cleared TTT fast weights, preserved SWA history")

        chunk_loger = images_loger[start:end]
        chunk_full = images_full[start:end] if images_full is not None else None

        # ---- Stage A ----
        print(f"\n  Stage A: Geometry Backbone ...")
        t0 = time.time()
        cache_ttt_primitives = args.ttt_write_mode != "native"
        stage_a_result = backbone.run(
            chunk_loger,
            ttt_state=ttt_state,
            cache_ttt_primitives=cache_ttt_primitives,
        )
        if cache_ttt_primitives:
            geo, write_cache = stage_a_result
        else:
            geo = stage_a_result
            write_cache = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"    done in {time.time() - t0:.2f}s")
        print_geometry_output(geo)
        all_geo.append(geo)

        # ---- Stage B ----
        cue: Optional[CueOutput] = None
        if not args.geometry_eval_mode:
            print(f"  Stage B: Dynamic Cue Extractor ...")
            t0 = time.time()
            assert extractor is not None
            cue = extractor.run(geo)
            print(f"    done in {time.time() - t0:.2f}s")
            print_cue_output(cue)
        else:
            print("  Stage B: skipped (--geometry_eval_mode)")
        all_cue.append(cue)

        # ---- Stage C ----
        mo: Optional[MaskletOutput] = None
        if not args.geometry_eval_mode:
            print(f"  Stage C: Video Masklet Front-end ...")
            t0 = time.time()
            assert chunk_full is not None
            if args.stage_memory_swap and str(args.device).startswith("cuda"):
                _offload_backbone_to_cpu(backbone)
            mo = _run_stage_c_lazy(build_kwargs, frontend_kwargs, chunk_full, ci)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"    done in {time.time() - t0:.2f}s")
            print_masklet_output(mo)
            if args.stage_memory_swap and str(args.device).startswith("cuda"):
                _offload_stage_c_frontend_to_cpu()
        else:
            print("  Stage C: skipped (--geometry_eval_mode)")
        all_masklet.append(mo)

        # ---- Stage D ----
        prior: Optional[PriorOutput] = None
        if need_semantic_prior:
            print(f"  Stage D: Semantic Prior Generator ...")
            t0 = time.time()
            assert prior_gen is not None
            prior = prior_gen.run(cue, mo, geo)
            print(f"    done in {time.time() - t0:.2f}s")
            print_prior_output(prior)
        else:
            print(f"  Stage D: skipped (ttt_write_mode={args.ttt_write_mode})")
        all_prior.append(prior)

        # ---- Stage E ----
        if args.ttt_write_mode == "native":
            ttt_state = backbone.get_ttt_state()
            if args.stage_memory_swap and str(args.device).startswith("cuda") and ttt_state is not None:
                ttt_state = _move_tensor_tree_to_device(ttt_state, "cpu")
            print("  Stage E: skipped (native write-through from Stage A state)")
        elif write_cache is not None and write_cache.num_ttt_layers > 0:
            print(f"  Stage E: TTT Write Controller ...")
            t0 = time.time()
            wr = controller.run(
                write_cache,
                prior.A_tok if prior is not None else None,
                B_chunk_geo=prior.B_chunk_geo if prior is not None else None,
                device=args.device,
            )
            print(f"    done in {time.time() - t0:.2f}s")
            print_write_result(wr)

            ttt_state = {"w0": wr.w0, "w1": wr.w1, "w2": wr.w2}
            if wr.history is not None:
                ttt_state["history"] = wr.history
            if args.stage_memory_swap and str(args.device).startswith("cuda"):
                ttt_state = _move_tensor_tree_to_device(ttt_state, "cpu")
        else:
            print("  Stage E: skipped (no TTT layers cached)")

        del geo, cue, write_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_geo_for_merge = align_chunk_geometry_outputs(all_geo, args.chunk_overlap)

    merged_local_points = merge_chunk_tensor_tail_trim(
        [g.local_points for g in all_geo_for_merge], args.chunk_overlap,
    )
    merged_world_points = merge_chunk_tensor_tail_trim(
        [g.world_points for g in all_geo_for_merge], args.chunk_overlap,
    )
    merged_camera_poses = merge_chunk_tensor_tail_trim(
        [g.camera_poses for g in all_geo_for_merge], args.chunk_overlap,
    )
    merged_confidence = merge_chunk_tensor_tail_trim(
        [g.confidence for g in all_geo_for_merge], args.chunk_overlap,
    )

    if args.output_txt and merged_camera_poses is not None:
        timestamps = build_timestamps_for_output(image_paths, args.input)
        twc = merged_camera_poses[:, :3, 3]
        qwc = mat_to_quat(merged_camera_poses[:, :3, :3])
        S = min(len(timestamps), twc.shape[0], qwc.shape[0])
        write_trajectory_txt(
            Path(args.output_txt),
            timestamps[:S],
            twc[:S].tolist(),
            qwc[:S].tolist(),
        )
        print(f"Saved trajectory to {args.output_txt}")

    # ===================================================================
    # Visualisation: use first chunk results for now (single-chunk mode)
    # ===================================================================
    if (
        all_masklet and all_cue
        and all_masklet[0] is not None
        and all_cue[0] is not None
    ):
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

    if args.output_pt and args.geometry_eval_mode:
        os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
        save_dict: Dict[str, torch.Tensor] = {}
        if merged_local_points is not None:
            save_dict["local_points"] = merged_local_points
        if merged_world_points is not None:
            save_dict["points"] = merged_world_points
        if merged_camera_poses is not None:
            save_dict["camera_poses"] = merged_camera_poses
        if merged_confidence is not None:
            save_dict["conf"] = merged_confidence
        torch.save(save_dict, args.output_pt)
        print(f"Saved output to {args.output_pt}")

    elif args.output_pt and all_masklet and all_masklet[0] is not None:
        os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
        mo = all_masklet[0]
        save_dict = {
            "M_mask": mo.M_mask,
            "V_mask": mo.V_mask,
            "L_sem": mo.L_sem,
            "G_sem": mo.G_sem,
            "W_sem": mo.W_sem,
        }
        if all_prior and all_prior[0] is not None:
            save_dict["A_tok"] = all_prior[0].A_tok
            save_dict["A_pix"] = all_prior[0].A_pix
            save_dict["Elig_pix"] = all_prior[0].Elig_pix
            save_dict["A_mask"] = all_prior[0].A_mask
            save_dict["r_mask"] = all_prior[0].r_mask
            save_dict["B_chunk_geo"] = torch.tensor(all_prior[0].B_chunk_geo)
        torch.save(save_dict, args.output_pt)
        print(f"Saved output to {args.output_pt}")

    import shutil
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)


# ---------------------------------------------------------------------------
# Lazy Stage C builder (so SAM2 + detector are only loaded when needed)
# ---------------------------------------------------------------------------
_frontend_instance: Optional[VideoMaskletFrontend] = None


def _move_module_to_device(module: Any, device: str) -> None:
    if module is not None and hasattr(module, "to"):
        module.to(device)


def _move_tensor_tree_to_device(obj: Any, device: str) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, list):
        return [_move_tensor_tree_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_tensor_tree_to_device(v, device) for v in obj)
    if isinstance(obj, dict):
        return {k: _move_tensor_tree_to_device(v, device) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        for field in fields(obj):
            try:
                value = _move_tensor_tree_to_device(getattr(obj, field.name), device)
                setattr(obj, field.name, value)
            except Exception:
                pass
        return obj
    return obj


def _ensure_backbone_on_device(backbone: LoGeRGeometryBackbone, device: str) -> None:
    _move_module_to_device(getattr(backbone, "model", None), device)


def _offload_backbone_to_cpu(backbone: LoGeRGeometryBackbone) -> None:
    _move_module_to_device(getattr(backbone, "model", None), "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _ensure_stage_c_frontend_on_device(device: str) -> None:
    if _frontend_instance is None:
        return
    predictor = getattr(_frontend_instance, "video_predictor", None)
    _move_module_to_device(getattr(predictor, "model", predictor), device)
    _move_module_to_device(getattr(_frontend_instance, "efficient_model", None), device)
    _move_module_to_device(getattr(_frontend_instance, "cutie_model", None), device)
    processor = getattr(_frontend_instance, "sam3_image_processor", None)
    if processor is not None:
        _move_module_to_device(getattr(processor, "model", None), device)
        if hasattr(processor, "device"):
            processor.device = device
        if hasattr(processor, "find_stage"):
            processor.find_stage = _move_tensor_tree_to_device(processor.find_stage, device)


def _offload_stage_c_frontend_to_cpu() -> None:
    if _frontend_instance is None:
        return
    detector = getattr(_frontend_instance, "detector", None)
    if detector is not None and hasattr(detector, "release_gpu"):
        detector.release_gpu()
    predictor = getattr(_frontend_instance, "video_predictor", None)
    _move_module_to_device(getattr(predictor, "model", predictor), "cpu")
    _move_module_to_device(getattr(_frontend_instance, "efficient_model", None), "cpu")
    _move_module_to_device(getattr(_frontend_instance, "cutie_model", None), "cpu")
    processor = getattr(_frontend_instance, "sam3_image_processor", None)
    if processor is not None:
        _move_module_to_device(getattr(processor, "model", None), "cpu")
        if hasattr(processor, "device"):
            processor.device = "cpu"
        if hasattr(processor, "find_stage"):
            processor.find_stage = _move_tensor_tree_to_device(processor.find_stage, "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_stage_c_lazy(
    build_kwargs: dict, frontend_kwargs: dict,
    chunk_images: torch.Tensor, chunk_idx: int,
) -> MaskletOutput:
    global _frontend_instance
    if _frontend_instance is None:
        print(
            "    Building VideoMaskletFrontend "
            f"(sam_backend={build_kwargs.get('sam_backend','?')}, "
            f"detector={build_kwargs.get('detector_type','?')}) ..."
        )
        _frontend_instance = VideoMaskletFrontend.from_config(
            **build_kwargs, **frontend_kwargs,
        )
    else:
        _ensure_stage_c_frontend_on_device(str(build_kwargs.get("device", "cuda")))
    return _frontend_instance.run(chunk_images, chunk_index=chunk_idx)


if __name__ == "__main__":
    main()
