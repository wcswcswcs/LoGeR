#!/usr/bin/env python3
"""Run SAM3 text prompts as a narrow STUFF-only sparse smoke test.

This does not modify the main Stage C pipeline. It asks SAM3 for one or more
text prompts on selected prompt frames, propagates the selected objects through
the video chunk, unions the kept objects per prompt, and writes the existing
sparse_masklets_v1 format plus an overlay video and metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
SAM3_ROOT = REPO_ROOT / "third_party" / "sam3"
for path in (SAM3_ROOT, REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import coverage_stats, parse_contact_frames, track_stats  # noqa: E402
from run_video_masklet_front_end import SparseMaskletOutput, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end import _unpack_mask_np  # noqa: E402
from run_video_masklet_front_end_v2 import _make_track, _write_mask, create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 text prompts as sparse STUFF smoke.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam3_checkpoint", default="ckpts/SAM3/sam3.pt")
    parser.add_argument("--prompts", required=True, help="Pipe-separated SAM3 text prompts.")
    parser.add_argument("--label", default="curtain", help="Sparse semantic label to write.")
    parser.add_argument("--start_frame", type=int, default=120)
    parser.add_argument("--frames_limit", type=int, default=64)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--prompt_frames", default="0", help="Comma-separated local chunk frame indices.")
    parser.add_argument("--max_objects_per_prompt", type=int, default=4)
    parser.add_argument("--min_area_ratio", type=float, default=0.003)
    parser.add_argument("--max_area_ratio", type=float, default=0.75)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--union_same_label", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact_frames", default="0,10,20,30,40,50,63")
    return parser.parse_args()


def _split_pipe(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def _split_ints(value: str) -> List[int]:
    return [int(item.strip()) for item in str(value or "").split(",") if item.strip()]


def _load_processing_frames(args: argparse.Namespace) -> Tuple[List[str], List[str], Tuple[int, int]]:
    image_paths, temp_dir = collect_image_paths(
        args.input_video,
        max(int(args.start_frame), 0),
        -1,
        1,
    )
    temp_dirs = [temp_dir] if temp_dir else []
    if args.frames_limit and int(args.frames_limit) > 0:
        image_paths = image_paths[: int(args.frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    return list(image_paths), temp_dirs, (int(proc_shape[0]), int(proc_shape[1]))


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _object_candidates(outputs: Dict[str, Any], H: int, W: int, args: argparse.Namespace) -> List[Dict[str, Any]]:
    obj_ids = _as_numpy(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks", [])
    probs = _as_numpy(outputs.get("out_probs", []))
    if isinstance(masks, torch.Tensor):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim < 3 or len(obj_ids) == 0:
        return []

    candidates: List[Dict[str, Any]] = []
    for idx, obj_id in enumerate(obj_ids):
        mask = np.squeeze(masks[idx]).astype(bool)
        if mask.shape != (H, W):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        area_ratio = float(mask.sum()) / float(max(H * W, 1))
        score = float(probs[idx]) if idx < len(probs) else 0.0
        status = "keep"
        if area_ratio < float(args.min_area_ratio):
            status = "area_lt_min"
        elif area_ratio > float(args.max_area_ratio):
            status = "area_gt_max"
        elif score < float(args.min_score):
            status = "score_lt_min"
        candidates.append(
            {
                "idx": int(idx),
                "obj_id": int(obj_id),
                "score": score,
                "area_ratio": area_ratio,
                "status": status,
            }
        )

    kept = [item for item in candidates if item["status"] == "keep"]
    kept.sort(key=lambda item: (item["score"], item["area_ratio"]), reverse=True)
    keep_ids = {int(item["obj_id"]) for item in kept[: max(int(args.max_objects_per_prompt), 0)]}
    for item in candidates:
        if item["status"] == "keep" and int(item["obj_id"]) not in keep_ids:
            item["status"] = "rank_pruned"
    return candidates


def _collect_prompt_track(
    predictor: Any,
    resource_path: str,
    prompt_text: str,
    prompt_frame: int,
    label: str,
    H: int,
    W: int,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    session_id = None
    debug: Dict[str, Any] = {
        "prompt_text": prompt_text,
        "prompt_frame": int(prompt_frame),
        "status": "not_started",
        "prompt_candidates": [],
        "kept_obj_ids": [],
        "frames_written": 0,
    }
    try:
        response = predictor.handle_request(
            dict(type="start_session", resource_path=resource_path)
        )
        session_id = response["session_id"]
        response = predictor.handle_request(
            dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=int(prompt_frame),
                text=str(prompt_text),
            )
        )
        outputs = response.get("outputs", {})
        prompt_candidates = _object_candidates(outputs, H, W, args)
        debug["prompt_candidates"] = prompt_candidates
        kept_obj_ids = {
            int(item["obj_id"])
            for item in prompt_candidates
            if item.get("status") == "keep"
        }
        debug["kept_obj_ids"] = sorted(kept_obj_ids)
        if not kept_obj_ids:
            debug["status"] = "no_kept_prompt_objects"
            return None, debug

        track = _make_track(
            label=label,
            source_type="stuff_static",
            birth_frame=int(prompt_frame),
            H=H,
            W=W,
            refine_status="sam3_text_prompt",
            proposal_tracklet_id=None,
        )
        frame_scores: Dict[int, List[float]] = {}

        prompt_obj_ids = _as_numpy(outputs.get("out_obj_ids", []))
        prompt_masks = outputs.get("out_binary_masks", [])
        prompt_probs = _as_numpy(outputs.get("out_probs", []))
        if isinstance(prompt_masks, torch.Tensor):
            prompt_masks = prompt_masks.detach().cpu().numpy()
        for idx, obj_id in enumerate(prompt_obj_ids):
            if int(obj_id) not in kept_obj_ids:
                continue
            mask = np.squeeze(prompt_masks[idx]).astype(bool)
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if int(prompt_frame) not in track["mask_by_frame"]:
                _write_mask(
                    track,
                    int(prompt_frame),
                    mask,
                    float(prompt_probs[idx]) if idx < len(prompt_probs) else 1.0,
                    H,
                    W,
                )
                frame_scores[int(prompt_frame)] = [float(prompt_probs[idx]) if idx < len(prompt_probs) else 1.0]
            else:
                existing = track["mask_by_frame"][int(prompt_frame)]
                union = _unpack_mask_np(existing, H, W) | mask
                score = float(prompt_probs[idx]) if idx < len(prompt_probs) else 1.0
                frame_scores.setdefault(int(prompt_frame), []).append(score)
                _write_mask(track, int(prompt_frame), union, float(np.mean(frame_scores[int(prompt_frame)])), H, W)

        for prop_response in predictor.handle_stream_request(
            dict(type="propagate_in_video", session_id=session_id)
        ):
            frame_idx = int(prop_response.get("frame_index", -1))
            if frame_idx < 0:
                continue
            prop_outputs = prop_response.get("outputs", {})
            obj_ids = _as_numpy(prop_outputs.get("out_obj_ids", []))
            masks = prop_outputs.get("out_binary_masks", [])
            probs = _as_numpy(prop_outputs.get("out_probs", []))
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
            if np.asarray(masks).ndim < 3:
                continue
            union = np.zeros((H, W), dtype=bool)
            scores: List[float] = []
            for idx, obj_id in enumerate(obj_ids):
                if int(obj_id) not in kept_obj_ids:
                    continue
                mask = np.squeeze(masks[idx]).astype(bool)
                if mask.shape != (H, W):
                    mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                union |= mask
                scores.append(float(probs[idx]) if idx < len(probs) else 1.0)
            if union.any():
                _write_mask(track, frame_idx, union, float(np.mean(scores)) if scores else 1.0, H, W)

        debug["frames_written"] = int(len(track["mask_by_frame"]))
        debug["status"] = "written" if track["mask_by_frame"] else "empty_after_propagation"
        return (track if track["mask_by_frame"] else None), debug
    except Exception as exc:
        debug["status"] = "error"
        debug["error"] = repr(exc)
        raise
    finally:
        if session_id is not None:
            with contextlib.suppress(Exception):
                predictor.handle_request(dict(type="close_session", session_id=session_id))


def _union_tracks_same_label(tracks: Sequence[Dict[str, Any]], H: int, W: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for track in tracks:
        key = (str(track.get("L_sem", "")), str(track.get("source_type", "")))
        grouped.setdefault(key, []).append(track)

    merged_tracks: List[Dict[str, Any]] = []
    for (label, source_type), group in grouped.items():
        if len(group) == 1:
            merged_tracks.append(group[0])
            continue
        birth_frame = min(int(track.get("birth_frame", 0)) for track in group)
        merged = _make_track(
            label=label,
            source_type=source_type,
            birth_frame=birth_frame,
            H=H,
            W=W,
            refine_status="sam3_text_prompt_union_same_label",
            proposal_tracklet_id=None,
        )
        frames = sorted({int(f) for track in group for f in track.get("mask_by_frame", {}).keys()})
        for frame_idx in frames:
            union = np.zeros((H, W), dtype=bool)
            scores: List[float] = []
            for track in group:
                packed = track.get("mask_by_frame", {}).get(frame_idx)
                if packed is None:
                    continue
                union |= _unpack_mask_np(packed, H, W)
                scores.append(float(track.get("q_by_frame", {}).get(frame_idx, 1.0)))
            if union.any():
                _write_mask(merged, frame_idx, union, float(np.mean(scores)) if scores else 1.0, H, W)
        merged["_union_source_track_count"] = len(group)
        merged_tracks.append(merged)
    return merged_tracks


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = _split_pipe(args.prompts)
    if not prompts:
        raise ValueError("--prompts cannot be empty")
    prompt_frames = _split_ints(args.prompt_frames)
    if not prompt_frames:
        prompt_frames = [0]
    label = canonicalize_label(str(args.label))

    checkpoint = Path(args.sam3_checkpoint).expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    H, W = proc_shape
    if not image_paths:
        raise ValueError("No frames loaded from input video")
    resource_path = str(Path(image_paths[0]).parent)

    from sam3.model_builder import build_sam3_video_predictor  # noqa: E402

    if not str(args.device).startswith("cuda"):
        raise ValueError("This SAM3 runner expects a CUDA device.")
    device_index = int(str(args.device).split(":", 1)[1]) if ":" in str(args.device) else torch.cuda.current_device()
    torch.cuda.set_device(device_index)

    t0 = time.time()
    predictor = build_sam3_video_predictor(
        checkpoint_path=str(checkpoint),
        gpus_to_use=[device_index],
        async_loading_frames=False,
    )

    tracks: List[Dict[str, Any]] = []
    prompt_debug: List[Dict[str, Any]] = []
    try:
        with torch.inference_mode():
            for prompt_text in prompts:
                for prompt_frame in prompt_frames:
                    if prompt_frame < 0 or prompt_frame >= len(image_paths):
                        prompt_debug.append(
                            {
                                "prompt_text": prompt_text,
                                "prompt_frame": int(prompt_frame),
                                "status": "prompt_frame_out_of_range",
                            }
                        )
                        continue
                    track, debug = _collect_prompt_track(
                        predictor,
                        resource_path,
                        prompt_text,
                        int(prompt_frame),
                        label,
                        H,
                        W,
                        args,
                    )
                    prompt_debug.append(debug)
                    if track is not None:
                        track["_sam3_prompt_text"] = prompt_text
                        tracks.append(track)
    finally:
        if hasattr(predictor, "shutdown"):
            with contextlib.suppress(Exception):
                predictor.shutdown()

    raw_track_count = len(tracks)
    if int(args.union_same_label):
        tracks = _union_tracks_same_label(tracks, H, W)

    sparse = SparseMaskletOutput(
        tracks=tracks,
        num_masklets=len(tracks),
        num_frames=len(image_paths),
        frame_height=H,
        frame_width=W,
        debug={
            "sam3_text_stuff_sparse": {
                "input_video": str(args.input_video),
                "start_frame": int(args.start_frame),
                "frames_limit": int(args.frames_limit),
                "processing_max_side": int(args.processing_max_side),
                "resource_path": resource_path,
                "sam3_checkpoint": str(checkpoint),
                "prompts": prompts,
                "label": label,
                "prompt_frames": prompt_frames,
                "max_objects_per_prompt": int(args.max_objects_per_prompt),
                "min_area_ratio": float(args.min_area_ratio),
                "max_area_ratio": float(args.max_area_ratio),
                "min_score": float(args.min_score),
                "union_same_label": bool(int(args.union_same_label)),
                "raw_track_count": int(raw_track_count),
                "final_track_count": int(len(tracks)),
                "elapsed_sec": time.time() - t0,
                "prompt_debug": prompt_debug,
            }
        },
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_sheet.jpg"
    metrics_path = output_dir / "metrics_summary.json"
    debug_path = output_dir / "prompt_debug.json"

    save_sparse_output(output_pt, sparse)
    create_tracking_video_v2(
        image_paths,
        sparse,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    contact_frames = parse_contact_frames(args.contact_frames, sparse.num_frames)
    _make_single_contact(image_paths, sparse, contact_frames, contact_path, float(args.mask_alpha))
    debug_path.write_text(json.dumps(sparse.debug["sam3_text_stuff_sparse"], ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "prompt_debug": str(debug_path),
        "coverage": coverage_stats(sparse),
        "track_stats": track_stats(sparse),
        "elapsed_sec": float(sparse.debug["sam3_text_stuff_sparse"]["elapsed_sec"]),
        "num_tracks": len(tracks),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
