#!/usr/bin/env python3
"""Refine sparse stuff masks from an existing v2 sparse_masklets.pt.

This is an offline audit tool: it does not run detection or semantic models.
It loads a compact sparse_masklets_v1 file, reconstructs mask_by_frame tracks,
optionally refines stuff_static masks with local GrabCut, subtracts thing masks,
then saves a new sparse file, overlay video, metrics JSON, and contact sheet.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_video_masklet_front_end import (
    SparseMaskletOutput,
    _mask_to_box_np,
    _pack_mask_np,
    _unpack_mask_np,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import (
    create_tracking_video_v2,
    render_clean_frame,
    save_sparse_output,
)

PROVENANCE_TRACK_KEYS = (
    "mask_source",
    "proposal_source",
    "tracking_source",
    "label_source",
    "semantic_resolver",
    "sam3_status",
    "sam3_backend",
    "postprocess_history",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine stuff masks in a sparse masklet output.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--mode", choices=["grabcut", "sam2_image", "none"], default="grabcut")
    parser.add_argument("--labels", default="all", help="Comma-separated stuff labels to refine, or all.")
    parser.add_argument("--dilate_px", type=int, default=7)
    parser.add_argument("--erode_px", type=int, default=3)
    parser.add_argument("--grabcut_iters", type=int, default=1)
    parser.add_argument("--min_area_keep", type=float, default=0.35)
    parser.add_argument("--max_area_growth", type=float, default=1.08)
    parser.add_argument("--subtract_thing", type=int, default=1)
    parser.add_argument("--sam2_checkpoint", default="/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_device", default="cuda:0")
    parser.add_argument("--sam2_mask_logit_scale", type=float, default=10.0)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _packed_to_np(packed: Any) -> np.ndarray:
    if isinstance(packed, torch.Tensor):
        return packed.detach().cpu().numpy().astype(np.uint8, copy=False)
    return np.asarray(packed, dtype=np.uint8)


def load_sparse(path: Path) -> SparseMaskletOutput:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "sparse_masklets_v1":
        raise ValueError(f"Unsupported sparse format: {payload.get('format')}")

    H = int(payload["frame_height"])
    W = int(payload["frame_width"])
    tracks: List[Dict[str, Any]] = []
    for raw_track in payload["tracks"]:
        frames = [int(x) for x in raw_track.get("frames", [])]
        track: Dict[str, Any] = {
            "mask_by_frame": {},
            "box_by_frame": {},
            "q_by_frame": {},
            "area_by_frame": {},
            "L_sem": raw_track["L_sem"],
            "G_sem": int(raw_track["G_sem"]),
            "W_sem": float(raw_track["W_sem"]),
            "source_type": raw_track["source_type"],
            "birth_frame": int(raw_track.get("birth_frame", frames[0] if frames else 0)),
            "frame_height": H,
            "frame_width": W,
        }
        for key in PROVENANCE_TRACK_KEYS:
            if key in raw_track:
                track[key] = raw_track[key]
        packed_masks = raw_track.get("packed_masks", [])
        boxes = raw_track.get("boxes", torch.zeros(0, 4, dtype=torch.float32))
        scores = raw_track.get("scores", torch.zeros(len(frames), dtype=torch.float32))
        area_ratio = raw_track.get("area_ratio", torch.zeros(len(frames), dtype=torch.float32))
        for idx, frame_idx in enumerate(frames):
            packed = _packed_to_np(packed_masks[idx])
            track["mask_by_frame"][frame_idx] = packed
            track["box_by_frame"][frame_idx] = boxes[idx].detach().cpu().float() if len(boxes) else torch.zeros(4)
            track["q_by_frame"][frame_idx] = float(scores[idx]) if idx < len(scores) else 1.0
            track["area_by_frame"][frame_idx] = float(area_ratio[idx]) if idx < len(area_ratio) else 0.0
        tracks.append(track)

    return SparseMaskletOutput(
        tracks=tracks,
        num_masklets=len(tracks),
        num_frames=int(payload["num_frames"]),
        frame_height=H,
        frame_width=W,
        debug=dict(payload.get("debug", {})),
    )


def clone_sparse(sparse: SparseMaskletOutput) -> SparseMaskletOutput:
    tracks: List[Dict[str, Any]] = []
    for track in sparse.tracks:
        copied = {
            "mask_by_frame": dict(track.get("mask_by_frame", {})),
            "box_by_frame": dict(track.get("box_by_frame", {})),
            "q_by_frame": dict(track.get("q_by_frame", {})),
            "area_by_frame": dict(track.get("area_by_frame", {})),
            "L_sem": track.get("L_sem"),
            "G_sem": int(track.get("G_sem", 0)),
            "W_sem": float(track.get("W_sem", 0.0)),
            "source_type": track.get("source_type"),
            "birth_frame": int(track.get("birth_frame", 0)),
            "frame_height": int(track.get("frame_height", sparse.frame_height)),
            "frame_width": int(track.get("frame_width", sparse.frame_width)),
        }
        for key in PROVENANCE_TRACK_KEYS:
            if key in track:
                copied[key] = track[key]
        tracks.append(copied)
    return SparseMaskletOutput(
        tracks=tracks,
        num_masklets=len(tracks),
        num_frames=sparse.num_frames,
        frame_height=sparse.frame_height,
        frame_width=sparse.frame_width,
        debug=dict(sparse.debug),
    )


def _label_set(labels: str) -> Optional[set[str]]:
    if str(labels).strip().lower() == "all":
        return None
    return {x.strip().lower() for x in labels.split(",") if x.strip()}


def _kernel(px: int) -> np.ndarray:
    size = max(1, int(px))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _bbox_from_mask(mask: np.ndarray, pad: int, H: int, W: int) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask.astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(W - 1, int(xs.max()) + pad)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(H - 1, int(ys.max()) + pad)
    return x1, y1, x2, y2


def refine_one_mask_grabcut(
    rgb: np.ndarray,
    mask: np.ndarray,
    thing_mask: Optional[np.ndarray],
    dilate_px: int,
    erode_px: int,
    iters: int,
    min_area_keep: float,
    max_area_growth: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    H, W = mask.shape
    orig = mask.astype(bool)
    orig_area = int(orig.sum())
    if orig_area == 0:
        return orig, {"status": "empty_input", "orig_area": 0, "new_area": 0, "area_ratio": 0.0}

    dilate_kernel = _kernel(dilate_px)
    erode_kernel = _kernel(erode_px)
    allowed = cv2.dilate(orig.astype(np.uint8), dilate_kernel, iterations=1).astype(bool)
    bbox = _bbox_from_mask(allowed, pad=max(dilate_px * 2, 8), H=H, W=W)
    if bbox is None:
        return orig, {"status": "empty_allowed", "orig_area": orig_area, "new_area": orig_area, "area_ratio": 1.0}

    x1, y1, x2, y2 = bbox
    image_roi = rgb[y1 : y2 + 1, x1 : x2 + 1]
    orig_roi = orig[y1 : y2 + 1, x1 : x2 + 1]
    allowed_roi = allowed[y1 : y2 + 1, x1 : x2 + 1]
    thing_roi = thing_mask[y1 : y2 + 1, x1 : x2 + 1] if thing_mask is not None else None

    gc_mask = np.full(orig_roi.shape, cv2.GC_BGD, dtype=np.uint8)
    gc_mask[allowed_roi] = cv2.GC_PR_BGD
    gc_mask[orig_roi] = cv2.GC_PR_FGD
    eroded = cv2.erode(orig_roi.astype(np.uint8), erode_kernel, iterations=1).astype(bool)
    if eroded.any():
        gc_mask[eroded] = cv2.GC_FGD
    if thing_roi is not None and thing_roi.any():
        gc_mask[thing_roi.astype(bool)] = cv2.GC_BGD

    try:
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        image_bgr = cv2.cvtColor(image_roi, cv2.COLOR_RGB2BGR)
        cv2.grabCut(
            image_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            max(1, int(iters)),
            cv2.GC_INIT_WITH_MASK,
        )
        refined_roi = np.logical_or(gc_mask == cv2.GC_FGD, gc_mask == cv2.GC_PR_FGD)
    except Exception as exc:
        return orig, {
            "status": "grabcut_error",
            "error": str(exc),
            "orig_area": orig_area,
            "new_area": orig_area,
            "area_ratio": 1.0,
        }

    refined_roi &= allowed_roi
    if thing_roi is not None:
        refined_roi &= ~thing_roi.astype(bool)

    refined = np.zeros_like(orig, dtype=bool)
    refined[y1 : y2 + 1, x1 : x2 + 1] = refined_roi
    refined = cv2.morphologyEx(refined.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(3), iterations=1).astype(bool)
    if thing_mask is not None:
        refined &= ~thing_mask.astype(bool)

    new_area = int(refined.sum())
    ratio = float(new_area) / float(max(orig_area, 1))
    if new_area == 0 or ratio < float(min_area_keep) or ratio > float(max_area_growth):
        return orig, {
            "status": "rejected_area_guard",
            "orig_area": orig_area,
            "new_area": new_area,
            "area_ratio": ratio,
        }

    changed = int(np.logical_xor(orig, refined).sum())
    return refined, {
        "status": "accepted",
        "orig_area": orig_area,
        "new_area": new_area,
        "area_ratio": ratio,
        "changed_pixels": changed,
    }


def build_sam2_image_predictor(checkpoint: str, model_cfg: str, device: str) -> Any:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = os.path.expanduser(str(checkpoint))
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")
    sam2_model = build_sam2(str(model_cfg), ckpt, device=str(device))
    return SAM2ImagePredictor(sam2_model)


def refine_one_mask_sam2_image(
    predictor: Any,
    mask: np.ndarray,
    thing_mask: Optional[np.ndarray],
    dilate_px: int,
    min_area_keep: float,
    max_area_growth: float,
    mask_logit_scale: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    H, W = mask.shape
    orig = mask.astype(bool)
    orig_area = int(orig.sum())
    if orig_area == 0:
        return orig, {"status": "empty_input", "orig_area": 0, "new_area": 0, "area_ratio": 0.0}

    box = _mask_to_box_np(orig)
    if float(box[2] - box[0]) < 2.0 or float(box[3] - box[1]) < 2.0:
        return orig, {"status": "tiny_box", "orig_area": orig_area, "new_area": orig_area, "area_ratio": 1.0}

    low_res = cv2.resize(orig.astype(np.float32), (256, 256), interpolation=cv2.INTER_LINEAR)
    mask_input = (low_res * 2.0 - 1.0) * float(mask_logit_scale)
    try:
        masks, scores, _logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
    except Exception as exc:
        return orig, {
            "status": "sam2_error",
            "error": str(exc),
            "orig_area": orig_area,
            "new_area": orig_area,
            "area_ratio": 1.0,
        }

    out = np.asarray(masks)
    if out.ndim == 4:
        out = out.squeeze(1)
    if out.ndim == 3:
        out = out[0]
    refined = out.astype(bool)
    if refined.shape != orig.shape:
        refined = cv2.resize(refined.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    allowed = cv2.dilate(orig.astype(np.uint8), _kernel(dilate_px), iterations=1).astype(bool)
    refined &= allowed
    if thing_mask is not None:
        refined &= ~thing_mask.astype(bool)

    new_area = int(refined.sum())
    ratio = float(new_area) / float(max(orig_area, 1))
    score = float(np.asarray(scores).reshape(-1)[0]) if np.asarray(scores).size else 0.0
    if new_area == 0 or ratio < float(min_area_keep) or ratio > float(max_area_growth):
        return orig, {
            "status": "rejected_area_guard",
            "orig_area": orig_area,
            "new_area": new_area,
            "area_ratio": ratio,
            "sam2_score": score,
        }

    changed = int(np.logical_xor(orig, refined).sum())
    return refined, {
        "status": "accepted",
        "orig_area": orig_area,
        "new_area": new_area,
        "area_ratio": ratio,
        "changed_pixels": changed,
        "sam2_score": score,
    }


def build_thing_union(sparse: SparseMaskletOutput, frame_idx: int) -> np.ndarray:
    union = np.zeros((sparse.frame_height, sparse.frame_width), dtype=bool)
    for track in sparse.tracks:
        if str(track.get("source_type")) == "stuff_static":
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        union |= _unpack_mask_np(np.asarray(packed, dtype=np.uint8), sparse.frame_height, sparse.frame_width)
    return union


def refresh_track_frame(track: Dict[str, Any], frame_idx: int, mask: np.ndarray, H: int, W: int) -> None:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        track["mask_by_frame"].pop(frame_idx, None)
        track["box_by_frame"].pop(frame_idx, None)
        track["q_by_frame"].pop(frame_idx, None)
        track["area_by_frame"].pop(frame_idx, None)
        return
    track["mask_by_frame"][frame_idx] = _pack_mask_np(mask_bool)
    track["box_by_frame"][frame_idx] = torch.from_numpy(_mask_to_box_np(mask_bool))
    track["area_by_frame"][frame_idx] = float(mask_bool.sum()) / float(max(H * W, 1))


def coverage_stats(sparse: SparseMaskletOutput) -> Dict[str, Any]:
    H, W = sparse.frame_height, sparse.frame_width
    all_cov: List[float] = []
    stuff_cov: List[float] = []
    for frame_idx in range(sparse.num_frames):
        union = np.zeros((H, W), dtype=bool)
        stuff_union = np.zeros((H, W), dtype=bool)
        for track in sparse.tracks:
            packed = track.get("mask_by_frame", {}).get(frame_idx)
            if packed is None:
                continue
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
            union |= mask
            if str(track.get("source_type")) == "stuff_static":
                stuff_union |= mask
        all_cov.append(float(union.mean()))
        stuff_cov.append(float(stuff_union.mean()))
    return {
        "coverage_mean": float(np.mean(all_cov)) if all_cov else 0.0,
        "coverage_p10": float(np.percentile(all_cov, 10)) if all_cov else 0.0,
        "coverage_p50": float(np.percentile(all_cov, 50)) if all_cov else 0.0,
        "coverage_p90": float(np.percentile(all_cov, 90)) if all_cov else 0.0,
        "stuff_coverage_mean": float(np.mean(stuff_cov)) if stuff_cov else 0.0,
        "stuff_coverage_p10": float(np.percentile(stuff_cov, 10)) if stuff_cov else 0.0,
        "stuff_coverage_p50": float(np.percentile(stuff_cov, 50)) if stuff_cov else 0.0,
        "stuff_coverage_p90": float(np.percentile(stuff_cov, 90)) if stuff_cov else 0.0,
    }


def track_stats(sparse: SparseMaskletOutput) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, track in enumerate(sparse.tracks):
        areas = list(track.get("area_by_frame", {}).values())
        rows.append(
            {
                "track_index": idx,
                "label": track.get("L_sem"),
                "source_type": track.get("source_type"),
                "frames": len(track.get("mask_by_frame", {})),
                "mean_area_ratio": float(np.mean(areas)) if areas else 0.0,
                "max_area_ratio": float(np.max(areas)) if areas else 0.0,
            }
        )
    return rows


def parse_contact_frames(spec: str, limit: int) -> List[int]:
    frames: List[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if 0 <= idx < limit:
            frames.append(idx)
    return frames


def make_contact_sheet(
    image_paths: Sequence[str],
    before: SparseMaskletOutput,
    after: SparseMaskletOutput,
    frame_indices: Sequence[int],
    output_path: Path,
    mask_alpha: float,
) -> None:
    cells: List[np.ndarray] = []
    for frame_idx in frame_indices:
        bgr = cv2.imread(image_paths[frame_idx], cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((before.frame_height, before.frame_width, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (before.frame_height, before.frame_width):
                rgb = cv2.resize(rgb, (before.frame_width, before.frame_height))
        before_img = render_clean_frame(rgb, before, frame_idx, mask_alpha=mask_alpha)
        after_img = render_clean_frame(rgb, after, frame_idx, mask_alpha=mask_alpha)
        pair = np.concatenate([before_img, after_img], axis=1)
        cv2.putText(pair, f"frame {frame_idx}  before", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pair, "after", (before.frame_width + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(pair)
    if not cells:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact = np.concatenate(cells, axis=0)
    contact_bgr = cv2.cvtColor(contact, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), contact_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    refined = clone_sparse(sparse)
    allowed_labels = _label_set(args.labels)

    image_paths, temp_dir = collect_image_paths(args.input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if args.frames_limit and int(args.frames_limit) > 0:
        image_paths = image_paths[: int(args.frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(image_paths, int(args.processing_max_side))
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < sparse.num_frames:
        raise RuntimeError(f"Need at least {sparse.num_frames} frames, got {len(image_paths)}")
    image_paths = image_paths[: sparse.num_frames]
    if tuple(proc_shape) != (sparse.frame_height, sparse.frame_width):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(sparse.frame_height, sparse.frame_width)}")

    debug: Dict[str, Any] = {
        "input_pt": str(args.input_pt),
        "input_video": str(args.input_video),
        "mode": args.mode,
        "labels": args.labels,
        "dilate_px": int(args.dilate_px),
        "erode_px": int(args.erode_px),
        "grabcut_iters": int(args.grabcut_iters),
        "min_area_keep": float(args.min_area_keep),
        "max_area_growth": float(args.max_area_growth),
        "subtract_thing": int(args.subtract_thing),
        "sam2_checkpoint": str(args.sam2_checkpoint) if args.mode == "sam2_image" else "",
        "sam2_model_cfg": str(args.sam2_model_cfg) if args.mode == "sam2_image" else "",
        "sam2_device": str(args.sam2_device) if args.mode == "sam2_image" else "",
        "sam2_mask_logit_scale": float(args.sam2_mask_logit_scale) if args.mode == "sam2_image" else 0.0,
        "per_label": {},
    }

    before_stats = coverage_stats(sparse)
    if args.mode in {"grabcut", "sam2_image"}:
        H, W = sparse.frame_height, sparse.frame_width
        stuff_indices = [
            idx
            for idx, track in enumerate(refined.tracks)
            if str(track.get("source_type")) == "stuff_static"
            and (allowed_labels is None or str(track.get("L_sem", "")).lower() in allowed_labels)
        ]
        sam2_predictor = None
        if args.mode == "sam2_image":
            sam2_predictor = build_sam2_image_predictor(args.sam2_checkpoint, args.sam2_model_cfg, args.sam2_device)
        for frame_idx in range(sparse.num_frames):
            bgr = cv2.imread(image_paths[frame_idx], cv2.IMREAD_COLOR)
            if bgr is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[:2] != (H, W):
                    rgb = cv2.resize(rgb, (W, H))
            thing_union = build_thing_union(sparse, frame_idx) if int(args.subtract_thing) else None
            if sam2_predictor is not None:
                sam2_predictor.set_image(rgb)
            for idx in stuff_indices:
                track = refined.tracks[idx]
                packed = track.get("mask_by_frame", {}).get(frame_idx)
                if packed is None:
                    continue
                label = str(track.get("L_sem"))
                orig_mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
                if args.mode == "grabcut":
                    new_mask, row = refine_one_mask_grabcut(
                        rgb,
                        orig_mask,
                        thing_union,
                        int(args.dilate_px),
                        int(args.erode_px),
                        int(args.grabcut_iters),
                        float(args.min_area_keep),
                        float(args.max_area_growth),
                    )
                else:
                    assert sam2_predictor is not None
                    new_mask, row = refine_one_mask_sam2_image(
                        sam2_predictor,
                        orig_mask,
                        thing_union,
                        int(args.dilate_px),
                        float(args.min_area_keep),
                        float(args.max_area_growth),
                        float(args.sam2_mask_logit_scale),
                    )
                refresh_track_frame(track, frame_idx, new_mask, H, W)
                label_debug = debug["per_label"].setdefault(
                    label,
                    {"accepted": 0, "rejected_area_guard": 0, "errors": 0, "other": 0, "ratios": []},
                )
                status = str(row.get("status"))
                if status == "accepted":
                    label_debug["accepted"] += 1
                elif status == "rejected_area_guard":
                    label_debug["rejected_area_guard"] += 1
                elif status.endswith("error"):
                    label_debug["errors"] += 1
                else:
                    label_debug["other"] += 1
                label_debug["ratios"].append(float(row.get("area_ratio", 1.0)))

    for label, row in debug["per_label"].items():
        ratios = row.pop("ratios", [])
        row["mean_area_ratio_after_over_before"] = float(np.mean(ratios)) if ratios else 0.0
        row["p10_area_ratio_after_over_before"] = float(np.percentile(ratios, 10)) if ratios else 0.0
        row["p90_area_ratio_after_over_before"] = float(np.percentile(ratios, 90)) if ratios else 0.0

    after_stats = coverage_stats(refined)
    refined.debug["offline_stuff_boundary_refinement"] = debug

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_before_after.jpg"

    save_sparse_output(output_pt, refined)
    create_tracking_video_v2(
        list(image_paths),
        refined,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    contact_frames = parse_contact_frames(args.contact_frames, sparse.num_frames)
    make_contact_sheet(list(image_paths), sparse, refined, contact_frames, contact_path, float(args.mask_alpha))

    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": before_stats,
        "after": after_stats,
        "delta": {
            key: float(after_stats[key]) - float(before_stats[key])
            for key in before_stats.keys()
            if key in after_stats
        },
        "track_stats_after": track_stats(refined),
        "refinement_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
