#!/usr/bin/env python3
"""Run GroundingDINO text boxes + SAM2 image masks as a stuff-only sparse output.

This is an audit tool for indoor structure stuff labels such as wall, floor,
and ceiling. GroundingDINO proposes text-conditioned boxes per frame; SAM2 image
predictor turns those boxes into masks. The tool deliberately does not track
objects over time, so temporal quality can be judged directly from the saved
video/contact sheet.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    _make_sparse_stuff_track,
    _mask_to_box_np,
    _pack_mask_np,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402
from loger.pipeline.video_masklet_frontend import canonicalize_label  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO+SAM2 stuff-only sparse output.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=300)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling")
    parser.add_argument(
        "--negative_labels",
        default="",
        help="Labels to detect and subtract from output masks, e.g. 'curtain,person'.",
    )
    parser.add_argument(
        "--text_prompt",
        default="",
        help="GroundingDINO prompt. Empty means '<label>. <label>. ...'.",
    )
    parser.add_argument("--grounding_model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--max_boxes_per_label", type=int, default=2)
    parser.add_argument("--min_area_ratio", type=float, default=0.003)
    parser.add_argument("--max_area_ratio", type=float, default=0.85)
    parser.add_argument("--morph_open_px", type=int, default=0)
    parser.add_argument("--morph_close_px", type=int, default=0)
    parser.add_argument("--sam2_checkpoint", default="/home/tmp_datasets/weights/sam/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_model_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _resolve_device(requested: str) -> str:
    text = str(requested or "auto").strip().lower()
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def _load_processing_frames(args: argparse.Namespace) -> tuple[List[str], List[str], tuple[int, int]]:
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
    return list(image_paths), temp_dirs, tuple(int(x) for x in proc_shape)


def _default_prompt(labels: Iterable[str]) -> str:
    parts = [canonicalize_label(label) for label in labels if str(label).strip()]
    return " ".join(f"{label}." for label in parts)


def _map_grounding_label(raw_label: Any, allowed: List[str]) -> Optional[str]:
    text = canonicalize_label(str(raw_label or ""))
    if text in allowed:
        return text
    tokens = {canonicalize_label(token) for token in text.replace(".", " ").replace(",", " ").split()}
    for label in allowed:
        if label in tokens or label in text:
            return label
    return None


def _kernel(px: int) -> np.ndarray:
    size = max(int(px), 0)
    if size <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = size * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _postprocess_mask(mask: np.ndarray, open_px: int, close_px: int) -> np.ndarray:
    out = np.asarray(mask).astype(bool)
    if open_px > 0:
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, _kernel(open_px)).astype(bool)
    if close_px > 0:
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(close_px)).astype(bool)
    return out


def _mask_from_sam_output(mask_like: Any) -> Optional[np.ndarray]:
    if mask_like is None:
        return None
    if isinstance(mask_like, torch.Tensor):
        arr = mask_like.detach().float().cpu().numpy()
    else:
        arr = np.asarray(mask_like)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        return None
    if arr.ndim > 2:
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    return (arr > 0.0).astype(bool)


def _select_boxes_by_label(
    result: Dict[str, Any],
    labels: List[str],
    max_boxes_per_label: int,
) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    raw_boxes = result.get("boxes", [])
    raw_scores = result.get("scores", [])
    raw_labels = result.get("labels", [])
    for idx, raw_label in enumerate(raw_labels):
        label = _map_grounding_label(raw_label, labels)
        if label is None or idx >= len(raw_boxes):
            continue
        score = float(raw_scores[idx]) if idx < len(raw_scores) else 0.0
        box = raw_boxes[idx]
        if isinstance(box, torch.Tensor):
            box_arr = box.detach().float().cpu().numpy()
        else:
            box_arr = np.asarray(box, dtype=np.float32)
        if box_arr.size < 4:
            continue
        selected[label].append(
            {
                "box": box_arr.reshape(-1)[:4].astype(np.float32),
                "score": score,
                "raw_label": str(raw_label),
            }
        )
    for label in labels:
        selected[label].sort(key=lambda item: float(item["score"]), reverse=True)
        selected[label] = selected[label][: max(int(max_boxes_per_label), 1)]
    return selected


def _write_label_frame(
    track: Dict[str, Any],
    frame_idx: int,
    mask: np.ndarray,
    score: float,
    H: int,
    W: int,
) -> None:
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return
    track["mask_by_frame"][int(frame_idx)] = _pack_mask_np(mask_bool)
    track["box_by_frame"][int(frame_idx)] = torch.from_numpy(_mask_to_box_np(mask_bool))
    track["q_by_frame"][int(frame_idx)] = float(score)
    track["area_by_frame"][int(frame_idx)] = float(mask_bool.sum()) / float(max(H * W, 1))


def _cleanup_temp_dirs(temp_dirs: List[str]) -> None:
    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    args = parse_args()
    t0 = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_labels = [canonicalize_label(label) for label in _parse_csv(args.labels)]
    output_labels = [
        label
        for idx, label in enumerate(output_labels)
        if label and label not in output_labels[:idx]
    ]
    negative_labels = [canonicalize_label(label) for label in _parse_csv(args.negative_labels)]
    negative_labels = [
        label
        for idx, label in enumerate(negative_labels)
        if label and label not in negative_labels[:idx] and label not in output_labels
    ]
    detect_labels = list(output_labels) + list(negative_labels)
    if not output_labels:
        raise ValueError("--labels resolved to an empty label set")
    text_prompt = str(args.text_prompt or "").strip() or _default_prompt(detect_labels)
    device = _resolve_device(args.device)

    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    H, W = int(proc_shape[0]), int(proc_shape[1])
    tracks = {label: _make_sparse_stuff_track(label, H, W) for label in output_labels}
    debug: Dict[str, Any] = {
        "format": "grounded_sam2_stuff_sparse_v1",
        "grounding_model": str(args.grounding_model),
        "text_prompt": text_prompt,
        "labels": output_labels,
        "negative_labels": negative_labels,
        "detect_labels": detect_labels,
        "box_threshold": float(args.box_threshold),
        "text_threshold": float(args.text_threshold),
        "max_boxes_per_label": int(args.max_boxes_per_label),
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "sam2_checkpoint": str(args.sam2_checkpoint),
        "sam2_model_cfg": str(args.sam2_model_cfg),
        "device": device,
        "gsam2_root": str(GSAM2_ROOT),
        "frame_debug": [],
        "rejected_area": {},
        "negative_subtracted_pixels": {},
        "frames_with_any": 0,
    }

    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(args.grounding_model))
        grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(str(args.grounding_model)).to(device)
        grounding_model.eval()
        sam2_model = build_sam2(str(args.sam2_model_cfg), str(args.sam2_checkpoint), device=device)
        predictor = SAM2ImagePredictor(sam2_model)

        with torch.inference_mode():
            for frame_idx, image_path in enumerate(image_paths):
                image = Image.open(image_path).convert("RGB")
                rgb_np = np.asarray(image)
                inputs = processor(images=image, text=text_prompt, return_tensors="pt")
                inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
                outputs = grounding_model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs.get("input_ids"),
                    threshold=float(args.box_threshold),
                    text_threshold=float(args.text_threshold),
                    target_sizes=[(H, W)],
                )[0]
                selected = _select_boxes_by_label(result, detect_labels, int(args.max_boxes_per_label))

                predictor.set_image(rgb_np)
                frame_written = 0
                frame_item: Dict[str, Any] = {
                    "frame_idx": int(frame_idx),
                    "selected_boxes": {
                        label: [
                            {
                                "score": float(item["score"]),
                                "raw_label": str(item["raw_label"]),
                                "box": [float(v) for v in item["box"].tolist()],
                            }
                            for item in items
                        ]
                        for label, items in selected.items()
                        if items
                    },
                    "written": {},
                    "negative_subtracted": {},
                }
                label_masks: Dict[str, np.ndarray] = {}
                label_scores: Dict[str, float] = {}
                for label, items in selected.items():
                    if not items:
                        continue
                    boxes = np.stack([item["box"] for item in items], axis=0).astype(np.float32)
                    masks, mask_scores, _logits = predictor.predict(box=boxes, multimask_output=False)
                    union = np.zeros((H, W), dtype=bool)
                    score_values = [float(item["score"]) for item in items]
                    if mask_scores is not None:
                        score_values.extend(float(v) for v in np.asarray(mask_scores).reshape(-1).tolist())
                    for mask_like in np.asarray(masks):
                        mask = _mask_from_sam_output(mask_like)
                        if mask is not None:
                            union |= mask
                    union = _postprocess_mask(union, int(args.morph_open_px), int(args.morph_close_px))
                    if not union.any():
                        continue
                    label_masks[label] = union
                    label_scores[label] = float(np.mean(score_values)) if score_values else 0.0

                negative_union = np.zeros((H, W), dtype=bool)
                for negative_label in negative_labels:
                    negative_union |= label_masks.get(negative_label, np.zeros((H, W), dtype=bool))

                for label in output_labels:
                    union = label_masks.get(label)
                    if union is None:
                        continue
                    if negative_union.any():
                        before_pixels = int(union.sum())
                        union = union & ~negative_union
                        subtracted = before_pixels - int(union.sum())
                        if subtracted > 0:
                            debug["negative_subtracted_pixels"].setdefault(label, 0)
                            debug["negative_subtracted_pixels"][label] += int(subtracted)
                            frame_item["negative_subtracted"][label] = int(subtracted)
                    union = _postprocess_mask(union, int(args.morph_open_px), int(args.morph_close_px))
                    area_ratio = float(union.sum()) / float(max(H * W, 1))
                    if area_ratio < float(args.min_area_ratio) or area_ratio > float(args.max_area_ratio):
                        debug["rejected_area"].setdefault(label, 0)
                        debug["rejected_area"][label] += 1
                        continue
                    score = float(label_scores.get(label, 0.0))
                    _write_label_frame(tracks[label], frame_idx, union, score, H, W)
                    frame_written += 1
                    frame_item["written"][label] = {
                        "area_ratio": area_ratio,
                        "score": score,
                    }
                if frame_written:
                    debug["frames_with_any"] += 1
                debug["frame_debug"].append(frame_item)
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sparse_tracks = [track for track in tracks.values() if track.get("mask_by_frame")]
    sparse = SparseMaskletOutput(
        tracks=sparse_tracks,
        num_masklets=len(sparse_tracks),
        num_frames=len(image_paths),
        frame_height=H,
        frame_width=W,
        debug={"grounded_sam2_stuff": debug},
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    metrics_path = output_dir / "metrics_summary.json"
    contact_path = output_dir / "contact_sheet.jpg"

    save_sparse_output(output_pt, sparse)
    create_tracking_video_v2(
        image_paths,
        sparse,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    _make_single_contact(
        image_paths,
        sparse,
        parse_contact_frames(args.contact_frames, sparse.num_frames),
        contact_path,
        float(args.mask_alpha),
    )

    summary = {
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "coverage": coverage_stats(sparse),
        "track_stats": track_stats(sparse),
        "grounded_sam2_debug": {
            key: value
            for key, value in debug.items()
            if key != "frame_debug"
        },
        "frame_debug_path": str(output_dir / "frame_debug.json"),
        "elapsed_seconds": float(time.time() - t0),
    }
    (output_dir / "frame_debug.json").write_text(
        json.dumps(debug["frame_debug"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _cleanup_temp_dirs(temp_dirs)


if __name__ == "__main__":
    main()
