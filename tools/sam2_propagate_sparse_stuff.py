#!/usr/bin/env python3
"""Propagate existing sparse stuff seeds with SAM2 video predictor.

This is an audit/experiment tool. It does not classify pixels. Stuff labels
come from an existing sparse_masklets_v1 file; SAM2 only propagates those
seed masks through time.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    coverage_stats,
    load_sparse,
    make_contact_sheet,
    parse_contact_frames,
    track_stats,
)
from run_video_masklet_front_end import (  # noqa: E402
    SparseMaskletOutput,
    _mask_to_box_np,
    _pack_mask_np,
    _unpack_mask_np,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propagate sparse stuff seed masks with SAM2 video predictor.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling,curtain")
    parser.add_argument("--seed_frames", default="0")
    parser.add_argument("--min_seed_area_ratio", type=float, default=0.003)
    parser.add_argument("--max_seed_area_ratio", type=float, default=0.90)
    parser.add_argument("--min_output_area_ratio", type=float, default=0.0005)
    parser.add_argument("--max_output_area_ratio", type=float, default=0.90)
    parser.add_argument("--subtract_thing", type=int, default=1)
    parser.add_argument("--checkpoint", default="/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt")
    parser.add_argument("--model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offload_video_to_cpu", type=int, default=1)
    parser.add_argument("--offload_state_to_cpu", type=int, default=1)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _parse_csv_labels(spec: str) -> List[str]:
    labels = [item.strip().lower() for item in str(spec or "").split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required.")
    return labels


def _parse_seed_frames(spec: str, num_frames: int) -> List[int]:
    frames: List[int] = []
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        frame_idx = int(item)
        if 0 <= frame_idx < num_frames:
            frames.append(frame_idx)
    if not frames:
        raise ValueError(f"No valid seed frames in {spec!r} for {num_frames} frames.")
    return sorted(set(frames))


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> Tuple[List[str], List[str]]:
    image_paths, temp_dir = collect_image_paths(args.input_video, 0, -1, 1)
    temp_dirs = [temp_dir] if temp_dir else []
    if args.frames_limit and int(args.frames_limit) > 0:
        image_paths = image_paths[: int(args.frames_limit)]
    image_paths, resize_tmp, _orig_shape, proc_shape = prepare_processing_image_paths(
        image_paths,
        int(args.processing_max_side),
    )
    if resize_tmp:
        temp_dirs.append(resize_tmp)
    if len(image_paths) < num_frames:
        raise RuntimeError(f"Need at least {num_frames} frames, got {len(image_paths)}")
    image_paths = list(image_paths[:num_frames])
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return image_paths, temp_dirs


def _make_numeric_frame_dir(image_paths: Sequence[str], output_dir: Path) -> str:
    frame_dir = output_dir / "_sam2_numeric_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, src in enumerate(image_paths):
        dst = frame_dir / f"{idx:06d}.jpg"
        os.symlink(os.path.abspath(src), dst)
    return str(frame_dir)


def _build_label_track_map(sparse: SparseMaskletOutput, labels: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    wanted = {label.lower() for label in labels}
    tracks: Dict[str, Dict[str, Any]] = {}
    for track in sparse.tracks:
        if str(track.get("source_type")) != "stuff_static":
            continue
        label = str(track.get("L_sem", "")).lower()
        if label in wanted:
            tracks[label] = track
    return tracks


def _thing_union(sparse: SparseMaskletOutput, frame_idx: int) -> np.ndarray:
    H, W = sparse.frame_height, sparse.frame_width
    union = np.zeros((H, W), dtype=bool)
    for track in sparse.tracks:
        if str(track.get("source_type")) != "thing_tracked":
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            continue
        union |= _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
    return union


def _make_seed_masks(
    sparse: SparseMaskletOutput,
    label_tracks: Dict[str, Dict[str, Any]],
    labels: Sequence[str],
    label_to_obj: Dict[str, int],
    frame_idx: int,
    min_area: float,
    max_area: float,
    subtract_thing: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    H, W = sparse.frame_height, sparse.frame_width
    occupied = np.zeros((H, W), dtype=bool)
    thing = _thing_union(sparse, frame_idx) if subtract_thing else None
    masks: Dict[int, np.ndarray] = {}
    debug: Dict[str, Any] = {}
    for label in labels:
        track = label_tracks.get(label)
        if track is None:
            debug[label] = {"status": "missing_track"}
            continue
        packed = track.get("mask_by_frame", {}).get(int(frame_idx))
        if packed is None:
            debug[label] = {"status": "missing_seed_frame"}
            continue
        mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
        if thing is not None:
            mask &= ~thing
        mask &= ~occupied
        area = float(mask.mean())
        if area < min_area:
            debug[label] = {"status": "area_lt_min", "area_ratio": area}
            continue
        if area > max_area:
            debug[label] = {"status": "area_gt_max", "area_ratio": area}
            continue
        obj_id = int(label_to_obj[label])
        masks[obj_id] = mask.astype(bool)
        occupied |= mask
        debug[label] = {"status": "used", "area_ratio": area, "object_id": obj_id}
    return masks, debug


def _clone_thing_tracks(sparse: SparseMaskletOutput) -> List[Dict[str, Any]]:
    tracks: List[Dict[str, Any]] = []
    for track in sparse.tracks:
        if str(track.get("source_type")) != "thing_tracked":
            continue
        tracks.append(
            {
                "mask_by_frame": dict(track.get("mask_by_frame", {})),
                "box_by_frame": dict(track.get("box_by_frame", {})),
                "q_by_frame": dict(track.get("q_by_frame", {})),
                "area_by_frame": dict(track.get("area_by_frame", {})),
                "L_sem": track.get("L_sem"),
                "G_sem": int(track.get("G_sem", 0)),
                "W_sem": float(track.get("W_sem", 0.0)),
                "source_type": track.get("source_type"),
                "birth_frame": int(track.get("birth_frame", 0)),
                "frame_height": sparse.frame_height,
                "frame_width": sparse.frame_width,
            }
        )
    return tracks


def _make_empty_stuff_tracks(sparse: SparseMaskletOutput, labels: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    tracks: Dict[int, Dict[str, Any]] = {}
    templates = _build_label_track_map(sparse, labels)
    for obj_id, label in enumerate(labels, start=1):
        template = templates.get(label, {})
        tracks[obj_id] = {
            "mask_by_frame": {},
            "box_by_frame": {},
            "q_by_frame": {},
            "area_by_frame": {},
            "L_sem": label,
            "G_sem": int(template.get("G_sem", 0)),
            "W_sem": float(template.get("W_sem", 1.0)),
            "source_type": "stuff_static",
            "birth_frame": 0,
            "frame_height": sparse.frame_height,
            "frame_width": sparse.frame_width,
        }
    return tracks


def _write_mask(track: Dict[str, Any], frame_idx: int, mask: np.ndarray, score: float, H: int, W: int) -> None:
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return
    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask_bool)
    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask_bool))
    track["q_by_frame"][int(frame_idx)] = float(score)
    track["area_by_frame"][int(frame_idx)] = float(mask_bool.sum()) / float(max(H * W, 1))


def _run_segment(
    predictor: Any,
    video_dir: str,
    sparse: SparseMaskletOutput,
    label_tracks: Dict[str, Dict[str, Any]],
    labels: Sequence[str],
    label_to_obj: Dict[str, int],
    seed_frame: int,
    end_frame_exclusive: int,
    args: argparse.Namespace,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[str, Any]]:
    seed_masks, seed_debug = _make_seed_masks(
        sparse,
        label_tracks,
        labels,
        label_to_obj,
        seed_frame,
        float(args.min_seed_area_ratio),
        float(args.max_seed_area_ratio),
        bool(args.subtract_thing),
    )
    state = predictor.init_state(
        video_path=video_dir,
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        offload_state_to_cpu=bool(args.offload_state_to_cpu),
        async_loading_frames=False,
    )
    outputs: Dict[int, Dict[int, np.ndarray]] = {}
    try:
        for obj_id, mask in seed_masks.items():
            out_frame_idx, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                inference_state=state,
                frame_idx=int(seed_frame),
                obj_id=int(obj_id),
                mask=torch.from_numpy(mask.astype(np.float32)),
            )
            frame_outputs = outputs.setdefault(int(out_frame_idx), {})
            for idx, out_obj_id in enumerate(out_obj_ids):
                frame_outputs[int(out_obj_id)] = (out_mask_logits[idx] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
        max_frames = max(0, int(end_frame_exclusive) - int(seed_frame))
        if seed_masks and max_frames > 0:
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=int(seed_frame),
                max_frame_num_to_track=int(max_frames),
            ):
                if int(out_frame_idx) >= int(end_frame_exclusive):
                    continue
                frame_outputs = outputs.setdefault(int(out_frame_idx), {})
                for idx, obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[idx] > 0.0).detach().cpu().numpy().squeeze().astype(bool)
                    frame_outputs[int(obj_id)] = mask
    finally:
        try:
            predictor.reset_state(state)
        except Exception:
            pass
        try:
            state.clear()
        except Exception:
            pass
    return outputs, seed_debug


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    labels = _parse_csv_labels(args.labels)
    seed_frames = _parse_seed_frames(args.seed_frames, sparse.num_frames)
    image_paths, temp_dirs = _load_processing_frames(
        args,
        sparse.frame_height,
        sparse.frame_width,
        sparse.num_frames,
    )
    video_dir = _make_numeric_frame_dir(image_paths, output_dir)
    temp_dirs.append(video_dir)

    checkpoint = os.path.expanduser(str(args.checkpoint))
    if checkpoint and not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    if not str(args.model_cfg).strip():
        raise ValueError("SAM2 model cfg is empty.")

    from sam2.build_sam import build_sam2_video_predictor  # noqa: E402

    predictor = build_sam2_video_predictor(str(args.model_cfg), checkpoint, device=str(args.device))
    label_tracks = _build_label_track_map(sparse, labels)
    label_to_obj = {label: idx for idx, label in enumerate(labels, start=1)}
    obj_to_label = {idx: label for label, idx in label_to_obj.items()}
    stuff_tracks = _make_empty_stuff_tracks(sparse, labels)
    output_debug: Dict[str, Any] = {label: {"frames": 0, "dropped_too_small": 0, "dropped_too_large": 0} for label in labels}
    seed_debug: Dict[str, Any] = {}

    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for seg_idx, seed_frame in enumerate(seed_frames):
            end_frame = seed_frames[seg_idx + 1] if seg_idx + 1 < len(seed_frames) else sparse.num_frames
            outputs, debug = _run_segment(
                predictor,
                video_dir,
                sparse,
                label_tracks,
                labels,
                label_to_obj,
                seed_frame,
                end_frame,
                args,
            )
            seed_debug[str(seed_frame)] = debug
            for frame_idx, obj_masks in outputs.items():
                thing = _thing_union(sparse, frame_idx) if int(args.subtract_thing) else None
                for obj_id, mask in obj_masks.items():
                    label = obj_to_label.get(int(obj_id))
                    if label is None:
                        continue
                    if thing is not None:
                        mask = mask & ~thing
                    area = float(mask.mean())
                    if area < float(args.min_output_area_ratio):
                        output_debug[label]["dropped_too_small"] += 1
                        continue
                    if area > float(args.max_output_area_ratio):
                        output_debug[label]["dropped_too_large"] += 1
                        continue
                    _write_mask(stuff_tracks[int(obj_id)], frame_idx, mask, 1.0, sparse.frame_height, sparse.frame_width)
                    output_debug[label]["frames"] += 1
            print(
                f"SAM2 propagated segment {seg_idx + 1}/{len(seed_frames)} "
                f"seed={seed_frame} end={end_frame}",
                flush=True,
            )

    out_tracks = _clone_thing_tracks(sparse) + [
        track for obj_id, track in sorted(stuff_tracks.items()) if track["mask_by_frame"]
    ]
    output = SparseMaskletOutput(
        tracks=out_tracks,
        num_masklets=len(out_tracks),
        num_frames=sparse.num_frames,
        frame_height=sparse.frame_height,
        frame_width=sparse.frame_width,
        debug=dict(sparse.debug),
    )
    output.debug["offline_sam2_stuff_propagation"] = {
        "input_pt": str(args.input_pt),
        "labels": labels,
        "seed_frames": seed_frames,
        "label_to_object_id": label_to_obj,
        "min_seed_area_ratio": float(args.min_seed_area_ratio),
        "max_seed_area_ratio": float(args.max_seed_area_ratio),
        "min_output_area_ratio": float(args.min_output_area_ratio),
        "max_output_area_ratio": float(args.max_output_area_ratio),
        "subtract_thing": int(args.subtract_thing),
        "checkpoint": checkpoint,
        "model_cfg": str(args.model_cfg),
        "seed_debug": seed_debug,
        "output_debug": output_debug,
    }

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    metrics_path = output_dir / "metrics_summary.json"
    save_sparse_output(output_pt, output)
    create_tracking_video_v2(
        image_paths,
        output,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        sparse,
        output,
        parse_contact_frames(args.contact_frames, sparse.num_frames),
        contact_path,
        float(args.mask_alpha),
    )
    before = coverage_stats(sparse)
    after = coverage_stats(output)
    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": before,
        "after": after,
        "delta": {key: float(after[key]) - float(before[key]) for key in before.keys()},
        "track_stats_after": track_stats(output),
        "sam2_debug": output.debug["offline_sam2_stuff_propagation"],
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
