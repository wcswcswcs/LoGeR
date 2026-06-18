#!/usr/bin/env python3
"""Create sparse stuff seed masks from audited SAM2 image prompts.

This tool is intentionally small and explicit: prompts live in a JSON file with
frame indices, boxes, positive points, and negative points. It produces a
sparse_masklets_v1 file that can be inspected directly or propagated by
tools/sam2_propagate_sparse_stuff.py.
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

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
GSAM2_ROOT = Path(os.environ.get("GSAM2_ROOT", str(REPO_ROOT / "Grounded-SAM-2"))).resolve()
for path in (GSAM2_ROOT, REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import coverage_stats, parse_contact_frames, track_stats  # noqa: E402
from run_video_masklet_front_end import (  # noqa: E402
    SparseMaskletOutput,
    _mask_to_box_np,
    _pack_mask_np,
    _make_sparse_stuff_track,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM2 image prompt keyframes into a sparse stuff seed.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--prompts_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--label", default="", help="Override label from JSON.")
    parser.add_argument("--frames_limit", type=int, default=300)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--sam2_checkpoint", default="/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--multimask_output", type=int, default=1)
    parser.add_argument("--min_area_ratio", type=float, default=0.001)
    parser.add_argument("--max_area_ratio", type=float, default=0.80)
    parser.add_argument("--morph_open_px", type=int, default=0)
    parser.add_argument("--morph_close_px", type=int, default=0)
    parser.add_argument("--contact_frames", default="120,150,179,210,240,270,299")
    return parser.parse_args()


def _load_processing_frames(args: argparse.Namespace) -> Tuple[List[str], List[str], Tuple[int, int]]:
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
    return list(image_paths), temp_dirs, (int(proc_shape[0]), int(proc_shape[1]))


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _postprocess(mask: np.ndarray, open_px: int, close_px: int) -> np.ndarray:
    out = np.asarray(mask).astype(bool)
    if int(open_px) > 0 and out.any():
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _kernel(open_px)).astype(bool)
    if int(close_px) > 0 and out.any():
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(close_px)).astype(bool)
    return out


def _points(prompt: Dict[str, Any]) -> Tuple[np.ndarray | None, np.ndarray | None]:
    coords: List[List[float]] = []
    labels: List[int] = []
    for point in prompt.get("positive_points", []) or []:
        coords.append([float(point[0]), float(point[1])])
        labels.append(1)
    for point in prompt.get("negative_points", []) or []:
        coords.append([float(point[0]), float(point[1])])
        labels.append(0)
    if not coords:
        return None, None
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _box(prompt: Dict[str, Any]) -> np.ndarray | None:
    raw = prompt.get("box")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"Invalid box in prompt: {raw!r}")
    return arr


def _mask_from_output(mask_like: Any) -> np.ndarray | None:
    arr = np.asarray(mask_like)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        return None
    if arr.ndim > 2:
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    return (arr > 0.0).astype(bool)


def _choose_mask(
    masks: Any,
    scores: Any,
    min_area: float,
    max_area: float,
) -> Tuple[np.ndarray | None, float, Dict[str, Any]]:
    mask_arr = np.asarray(masks)
    if mask_arr.ndim == 2:
        mask_arr = mask_arr[None, ...]
    score_arr = np.asarray(scores if scores is not None else np.zeros((mask_arr.shape[0],), dtype=np.float32)).reshape(-1)
    candidates: List[Tuple[float, int, np.ndarray, float]] = []
    debug_candidates: List[Dict[str, Any]] = []
    for idx, mask_like in enumerate(mask_arr):
        mask = _mask_from_output(mask_like)
        if mask is None:
            continue
        area = float(mask.mean())
        score = float(score_arr[idx]) if idx < len(score_arr) else 0.0
        status = "candidate"
        if area < float(min_area):
            status = "area_lt_min"
        elif area > float(max_area):
            status = "area_gt_max"
        else:
            candidates.append((score, idx, mask, area))
        debug_candidates.append({"idx": int(idx), "score": score, "area_ratio": area, "status": status})
    if not candidates:
        return None, 0.0, {"candidates": debug_candidates, "chosen": None}
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, idx, mask, area = candidates[0]
    return mask, score, {"candidates": debug_candidates, "chosen": {"idx": int(idx), "score": score, "area_ratio": area}}


def _write_mask(track: Dict[str, Any], frame_idx: int, mask: np.ndarray, score: float, H: int, W: int) -> None:
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return
    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask_bool)
    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask_bool))
    track["q_by_frame"][int(frame_idx)] = float(score)
    track["area_by_frame"][int(frame_idx)] = float(mask_bool.sum()) / float(max(H * W, 1))


def _draw_prompt_contact(
    image_paths: Sequence[str],
    sparse: SparseMaskletOutput,
    prompts_by_frame: Dict[int, List[Dict[str, Any]]],
    frames: Sequence[int],
    output_path: Path,
    mask_alpha: float,
) -> None:
    cells: List[np.ndarray] = []
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    for frame_idx in frames:
        if frame_idx < 0 or frame_idx >= len(image_paths):
            continue
        bgr = cv2.imread(image_paths[frame_idx], cv2.IMREAD_COLOR)
        if bgr is None:
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H))
        prompt_img = rgb.copy()
        for prompt in prompts_by_frame.get(int(frame_idx), []):
            box = _box(prompt)
            if box is not None:
                x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
                cv2.rectangle(prompt_img, (x0, y0), (x1, y1), (255, 50, 50), 2)
            for point in prompt.get("positive_points", []) or []:
                cv2.circle(prompt_img, (int(round(point[0])), int(round(point[1]))), 5, (20, 255, 20), -1)
            for point in prompt.get("negative_points", []) or []:
                cv2.circle(prompt_img, (int(round(point[0])), int(round(point[1]))), 5, (255, 40, 40), -1)
        overlay = prompt_img.copy()
        rendered = sparse
        # Reuse renderer via the slice helper contact by making a one-frame cell manually.
        from run_video_masklet_front_end_v2 import render_clean_frame

        mask_img = render_clean_frame(rgb, rendered, frame_idx, mask_alpha=mask_alpha)
        pair = np.concatenate([prompt_img, mask_img], axis=1)
        cv2.putText(pair, f"frame {frame_idx} prompt", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pair, "sam2 seed", (W + 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(pair)
    if not cells:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contact = np.concatenate(cells, axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_data = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    label = canonicalize_label(str(args.label or prompts_data.get("label") or "opening"))
    prompts = list(prompts_data.get("prompts", []))
    if not prompts:
        raise ValueError("prompts_json contains no prompts")

    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    H, W = proc_shape
    track = _make_sparse_stuff_track(label, H, W)
    prompts_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for prompt in prompts:
        frame_idx = int(prompt["frame"])
        if frame_idx < 0 or frame_idx >= len(image_paths):
            raise ValueError(f"Prompt frame {frame_idx} is outside loaded frames {len(image_paths)}")
        prompts_by_frame.setdefault(frame_idx, []).append(prompt)

    from sam2.build_sam import build_sam2  # noqa: E402
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402

    checkpoint = os.path.expanduser(str(args.sam2_checkpoint))
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")
    sam2_model = build_sam2(str(args.sam2_model_cfg), checkpoint, device=str(args.device))
    predictor = SAM2ImagePredictor(sam2_model)

    debug: Dict[str, Any] = {
        "format": "sam2_prompt_sparse_v1",
        "input_video": str(args.input_video),
        "prompts_json": str(args.prompts_json),
        "label": label,
        "sam2_checkpoint": checkpoint,
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "multimask_output": int(args.multimask_output),
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "morph_open_px": int(args.morph_open_px),
        "morph_close_px": int(args.morph_close_px),
        "prompt_results": [],
    }

    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        for frame_idx in sorted(prompts_by_frame):
            image = Image.open(image_paths[frame_idx]).convert("RGB")
            rgb = np.asarray(image)
            predictor.set_image(rgb)
            union = np.zeros((H, W), dtype=bool)
            scores: List[float] = []
            frame_debug: Dict[str, Any] = {"frame_idx": int(frame_idx), "prompts": []}
            for prompt_idx, prompt in enumerate(prompts_by_frame[frame_idx]):
                point_coords, point_labels = _points(prompt)
                box = _box(prompt)
                masks, mask_scores, _logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=bool(int(args.multimask_output)),
                )
                mask, score, choose_debug = _choose_mask(
                    masks,
                    mask_scores,
                    float(args.min_area_ratio),
                    float(args.max_area_ratio),
                )
                item = {
                    "prompt_idx": int(prompt_idx),
                    "box": None if box is None else [float(v) for v in box.tolist()],
                    "positive_points": prompt.get("positive_points", []) or [],
                    "negative_points": prompt.get("negative_points", []) or [],
                    "choose": choose_debug,
                }
                if mask is not None:
                    union |= mask
                    scores.append(float(score))
                    item["status"] = "used"
                else:
                    item["status"] = "no_valid_mask"
                frame_debug["prompts"].append(item)
            union = _postprocess(union, int(args.morph_open_px), int(args.morph_close_px))
            if union.any():
                _write_mask(track, frame_idx, union, float(np.mean(scores)) if scores else 1.0, H, W)
                frame_debug["union_area_ratio"] = float(union.mean())
                frame_debug["status"] = "written"
            else:
                frame_debug["union_area_ratio"] = 0.0
                frame_debug["status"] = "empty_union"
            debug["prompt_results"].append(frame_debug)

    tracks = [track] if track.get("mask_by_frame") else []
    sparse = SparseMaskletOutput(
        tracks=tracks,
        num_masklets=len(tracks),
        num_frames=len(image_paths),
        frame_height=H,
        frame_width=W,
        debug={"sam2_prompt_sparse": debug},
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_sheet.jpg"
    prompt_contact_path = output_dir / "prompt_contact.jpg"
    metrics_path = output_dir / "metrics_summary.json"
    prompt_debug_path = output_dir / "prompt_debug.json"

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
    _draw_prompt_contact(image_paths, sparse, prompts_by_frame, contact_frames, prompt_contact_path, float(args.mask_alpha))
    prompt_debug_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "prompt_contact": str(prompt_contact_path),
        "prompt_debug": str(prompt_debug_path),
        "coverage": coverage_stats(sparse),
        "track_stats": track_stats(sparse),
        "sam2_prompt_debug": {
            key: value for key, value in debug.items() if key != "prompt_results"
        },
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
