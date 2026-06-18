#!/usr/bin/env python3
"""Propagate existing semantic stuff seeds with Cutie video object segmentation.

This is an audit/experiment tool. It does not classify pixels itself. It uses
stuff masks from an existing sparse_masklets_v1 file as seed masks, then asks
Cutie to propagate those object masks through the video.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
CUTIE_ROOT = REPO_ROOT / "third_party" / "Cutie"
for path in (REPO_ROOT, TOOLS_ROOT, CUTIE_ROOT):
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
    parser = argparse.ArgumentParser(description="Propagate sparse stuff seed masks with Cutie.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling,curtain,door")
    parser.add_argument("--seed_frames", default="0")
    parser.add_argument("--min_seed_area_ratio", type=float, default=0.003)
    parser.add_argument("--max_seed_area_ratio", type=float, default=0.75)
    parser.add_argument("--min_output_area_ratio", type=float, default=0.0005)
    parser.add_argument("--max_output_area_ratio", type=float, default=0.85)
    parser.add_argument("--subtract_thing", type=int, default=1)
    parser.add_argument("--max_internal_size", type=int, default=480)
    parser.add_argument(
        "--complete_seed_with_previous",
        type=int,
        default=1,
        help="When a later seed frame lacks an initialized label, fill it from the previous Cutie output.",
    )
    parser.add_argument(
        "--reset_on_seed",
        type=int,
        default=0,
        help="Reset the Cutie processor at each seed frame instead of injecting new masks into existing memory.",
    )
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--contact_frames", default="0,10,20,30,40,50,60,63")
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


def _make_seed_mask(
    sparse: SparseMaskletOutput,
    label_tracks: Dict[str, Dict[str, Any]],
    labels: Sequence[str],
    label_to_obj: Dict[str, int],
    frame_idx: int,
    min_area: float,
    max_area: float,
    subtract_thing: bool,
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    H, W = sparse.frame_height, sparse.frame_width
    seed = np.zeros((H, W), dtype=np.uint8)
    thing = _thing_union(sparse, frame_idx) if subtract_thing else None
    objects: List[int] = []
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
        area = float(mask.mean())
        if area < min_area:
            debug[label] = {"status": "area_lt_min", "area_ratio": area}
            continue
        if area > max_area:
            debug[label] = {"status": "area_gt_max", "area_ratio": area}
            continue
        obj_id = int(label_to_obj[label])
        seed[(mask) & (seed == 0)] = obj_id
        if np.any(seed == obj_id):
            objects.append(obj_id)
            debug[label] = {"status": "used", "area_ratio": area, "object_id": obj_id}
        else:
            debug[label] = {"status": "fully_occluded_by_priority", "area_ratio": area, "object_id": obj_id}
    return seed, objects, debug


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
    for idx, label in enumerate(labels, start=1):
        template = templates.get(label, {})
        tracks[idx] = {
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

    from hydra.core.global_hydra import GlobalHydra  # noqa: E402
    from cutie.inference.inference_core import InferenceCore  # noqa: E402
    from cutie.utils.get_default_model import get_default_model  # noqa: E402

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    cutie = get_default_model()
    def new_processor() -> Any:
        proc = InferenceCore(cutie, cfg=cutie.cfg)
        proc.max_internal_size = int(args.max_internal_size)
        return proc

    processor = new_processor()
    device = next(cutie.parameters()).device

    label_tracks = _build_label_track_map(sparse, labels)
    label_to_obj = {label: idx for idx, label in enumerate(labels, start=1)}
    obj_to_label = {idx: label for label, idx in label_to_obj.items()}
    stuff_tracks = _make_empty_stuff_tracks(sparse, labels)
    seed_debug: Dict[str, Any] = {}
    output_debug: Dict[str, Any] = {label: {"frames": 0, "dropped_too_small": 0, "dropped_too_large": 0} for label in labels}
    H, W = sparse.frame_height, sparse.frame_width
    last_out_mask: Optional[np.ndarray] = None

    with torch.inference_mode():
        for frame_idx, image_path in enumerate(image_paths):
            image = to_tensor(Image.open(image_path).convert("RGB")).to(device=device, dtype=torch.float32)
            seed_tensor: Optional[torch.Tensor] = None
            objects: Optional[List[int]] = None
            if frame_idx in seed_frames:
                if int(args.reset_on_seed) and frame_idx != seed_frames[0]:
                    processor = new_processor()
                    last_out_mask = None
                seed_np, objects_np, debug = _make_seed_mask(
                    sparse,
                    label_tracks,
                    labels,
                    label_to_obj,
                    frame_idx,
                    float(args.min_seed_area_ratio),
                    float(args.max_seed_area_ratio),
                    bool(args.subtract_thing),
                )
                if (
                    last_out_mask is not None
                    and int(args.complete_seed_with_previous)
                    and frame_idx != seed_frames[0]
                ):
                    filled_from_previous: Dict[str, float] = {}
                    for label, obj_id in label_to_obj.items():
                        if int(obj_id) in objects_np:
                            continue
                        prev_mask = (last_out_mask == int(obj_id)) & (seed_np == 0)
                        if not prev_mask.any():
                            continue
                        seed_np[prev_mask] = int(obj_id)
                        objects_np.append(int(obj_id))
                        filled_from_previous[label] = float(prev_mask.mean())
                    if filled_from_previous:
                        debug["_filled_from_previous"] = filled_from_previous
                seed_debug[str(frame_idx)] = debug
                if objects_np:
                    objects_np = sorted(set(int(x) for x in objects_np))
                    seed_tensor = torch.from_numpy(seed_np).to(device=device, dtype=torch.long)
                    objects = [int(x) for x in objects_np]
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and str(device).startswith("cuda")):
                if seed_tensor is not None:
                    output_prob = processor.step(image, seed_tensor, objects=objects, force_permanent=True)
                else:
                    output_prob = processor.step(image)
            out_mask = processor.output_prob_to_mask(output_prob).detach().cpu().numpy().astype(np.uint8)
            last_out_mask = out_mask
            thing = _thing_union(sparse, frame_idx) if int(args.subtract_thing) else None
            for obj_id, label in obj_to_label.items():
                mask = out_mask == int(obj_id)
                if thing is not None:
                    mask &= ~thing
                area = float(mask.mean())
                if area < float(args.min_output_area_ratio):
                    output_debug[label]["dropped_too_small"] += 1
                    continue
                if area > float(args.max_output_area_ratio):
                    output_debug[label]["dropped_too_large"] += 1
                    continue
                _write_mask(stuff_tracks[int(obj_id)], frame_idx, mask, 1.0, H, W)
                output_debug[label]["frames"] += 1
            if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == len(image_paths):
                print(f"Cutie propagated {frame_idx + 1}/{len(image_paths)} frames", flush=True)

    out_tracks = _clone_thing_tracks(sparse) + [
        track for obj_id, track in sorted(stuff_tracks.items()) if track["mask_by_frame"]
    ]
    output = SparseMaskletOutput(
        tracks=out_tracks,
        num_masklets=len(out_tracks),
        num_frames=sparse.num_frames,
        frame_height=H,
        frame_width=W,
        debug=dict(sparse.debug),
    )
    output.debug["offline_cutie_stuff_propagation"] = {
        "input_pt": str(args.input_pt),
        "labels": labels,
        "seed_frames": seed_frames,
        "label_to_object_id": label_to_obj,
        "min_seed_area_ratio": float(args.min_seed_area_ratio),
        "max_seed_area_ratio": float(args.max_seed_area_ratio),
        "min_output_area_ratio": float(args.min_output_area_ratio),
        "max_output_area_ratio": float(args.max_output_area_ratio),
        "subtract_thing": int(args.subtract_thing),
        "max_internal_size": int(args.max_internal_size),
        "complete_seed_with_previous": int(args.complete_seed_with_previous),
        "reset_on_seed": int(args.reset_on_seed),
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
    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": coverage_stats(sparse),
        "after": coverage_stats(output),
        "delta": {
            key: float(coverage_stats(output)[key]) - float(coverage_stats(sparse)[key])
            for key in coverage_stats(sparse).keys()
        },
        "track_stats_after": track_stats(output),
        "cutie_debug": output.debug["offline_cutie_stuff_propagation"],
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
