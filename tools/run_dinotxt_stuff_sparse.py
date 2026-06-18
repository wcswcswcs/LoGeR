#!/usr/bin/env python3
"""Run DINOtxt patch-text similarity as a STUFF-only sparse backend.

This is an audit smoke tool. It does not run detector, SAM, or fusion stages.
It converts per-patch text similarity into v2-compatible sparse stuff tracks so
the same renderer and metrics helpers can be used.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import coverage_stats, parse_contact_frames, track_stats  # noqa: E402
from run_video_masklet_front_end import SparseMaskletOutput, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import _make_track, _write_mask, create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DINOtxt STUFF-only sparse smoke.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dinov3_root", default="/mnt/data/users/chengshun.wang/pjs/dinov3")
    parser.add_argument("--weights", default="/home/tmp_datasets/weights/dino/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth")
    parser.add_argument("--backbone_weights", default="/home/tmp_datasets/weights/dino/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument(
        "--bpe_path",
        default="/mnt/data/users/chengshun.wang/.cache/torch/hub/mhamilton723_FeatUp_main/featup/featurizers/maskclip/bpe_simple_vocab_16e6.txt.gz",
    )
    parser.add_argument("--frames_limit", type=int, default=64)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling,curtain")
    parser.add_argument(
        "--negative_labels",
        default="person,people,clothing,hair,guitar,chair,table,desk,furniture,cabinet,door,window,mirror,screen,monitor,object,black background",
    )
    parser.add_argument(
        "--prompt_templates",
        default="a photo of a {label}|a photo of {label}|indoor {label}",
        help="Pipe-separated prompt templates. Embeddings are averaged per label.",
    )
    parser.add_argument("--score_margin", type=float, default=0.0)
    parser.add_argument("--min_area_ratio", type=float, default=0.004)
    parser.add_argument("--max_area_ratio", type=float, default=0.90)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--contact_frames", default="0,10,20,30,40,50,63")
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _split_templates(text: str) -> List[str]:
    out = [item.strip() for item in str(text or "").split("|") if item.strip()]
    return out or ["a photo of a {label}"]


def _resolve_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(requested)


def _load_processing_frames(args: argparse.Namespace) -> tuple[List[str], List[str], tuple[int, int]]:
    image_paths, temp_dir = collect_image_paths(args.input_video, max(int(args.start_frame), 0), -1, 1)
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


def _kernel(size: int) -> np.ndarray | None:
    k = int(size)
    if k <= 1:
        return None
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _postprocess_mask(mask: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
    out = mask.astype(np.uint8)
    if kernel is not None:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel, iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out.astype(bool)


def _read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _make_transform(input_size: int) -> Any:
    size = int(input_size)
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((size, size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _label_text_features(model: Any, tokenizer: Any, labels: Sequence[str], templates: Sequence[str], device: str) -> torch.Tensor:
    features: List[torch.Tensor] = []
    for label in labels:
        prompts = [template.format(label=label) for template in templates]
        tokens = tokenizer.tokenize(prompts).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            text_features = model.encode_text(tokens, normalize=True)[:, 1024:]
        text_features = F.normalize(text_features.float(), p=2, dim=-1).mean(dim=0)
        features.append(F.normalize(text_features, p=2, dim=0))
    return torch.stack(features, dim=0)


def _upsample_patch_scores(scores: torch.Tensor, output_hw: tuple[int, int]) -> np.ndarray:
    patches, classes = scores.shape
    side = int(round(float(patches) ** 0.5))
    if side * side != patches:
        raise RuntimeError(f"Cannot infer square patch grid from {patches} patches")
    x = scores.transpose(0, 1).reshape(1, classes, side, side)
    x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
    return x[0].float().detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = _split_csv(args.labels)
    negative_labels = _split_csv(args.negative_labels)
    if not labels:
        raise ValueError("--labels cannot be empty")
    all_labels = labels + [label for label in negative_labels if label not in labels]
    templates = _split_templates(args.prompt_templates)
    device = _resolve_device(args.device)

    dinov3_root = Path(args.dinov3_root).resolve()
    if str(dinov3_root) not in sys.path:
        sys.path.insert(0, str(dinov3_root))

    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    H, W = proc_shape
    kernel = _kernel(int(args.morph_kernel))
    min_area = float(args.min_area_ratio)
    max_area = float(args.max_area_ratio)
    score_margin = float(args.score_margin)

    tracks: Dict[str, Dict[str, Any]] = {
        label: _make_track(label, "stuff_static", 0, H, W, "dinotxt_patch_similarity", None)
        for label in labels
    }
    observed_pixels = {label: 0 for label in all_labels}
    masks_added = {label: 0 for label in labels}
    frames_with_any = 0

    t0 = time.time()
    print(f"DINOtxt loading root={dinov3_root} device={device}", flush=True)
    model, tokenizer = torch.hub.load(
        str(dinov3_root),
        "dinov3_vitl16_dinotxt_tet1280d20h24l",
        source="local",
        weights=str(args.weights),
        backbone_weights=str(args.backbone_weights),
        bpe_path_or_url=str(args.bpe_path),
        trust_repo=True,
    )
    model.eval().to(device)
    text_features = _label_text_features(model, tokenizer, all_labels, templates, device)
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    transform = _make_transform(int(args.input_size))

    try:
        for frame_idx, path in enumerate(image_paths):
            rgb = _read_rgb(path)
            image = Image.fromarray(rgb)
            x = transform(image)[None].to(device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                _image_features, patch_tokens, _backbone_patch_tokens = model.encode_image_with_patch_tokens(x, normalize=True)
                patch_tokens = F.normalize(patch_tokens.float(), p=2, dim=-1)
                patch_scores = torch.einsum("bpd,cd->pc", patch_tokens, text_features.to(patch_tokens.device))
            scores = _upsample_patch_scores(patch_scores, (H, W))
            order = np.argsort(scores, axis=0)
            pred = order[-1]
            best = np.take_along_axis(scores, pred[None, :, :], axis=0)[0]
            second = np.take_along_axis(scores, order[-2][None, :, :], axis=0)[0] if scores.shape[0] > 1 else np.zeros_like(best)
            margin = best - second

            any_mask = False
            for label in all_labels:
                observed_pixels[label] += int((pred == label_to_idx[label]).sum())
            for label in labels:
                label_idx = label_to_idx[label]
                mask = pred == label_idx
                if score_margin > 0:
                    mask &= margin >= score_margin
                mask = _postprocess_mask(mask, kernel)
                area_ratio = float(mask.sum()) / float(max(H * W, 1))
                if area_ratio < min_area or area_ratio > max_area:
                    continue
                _write_mask(tracks[label], frame_idx, mask, float(best[mask].mean()) if mask.any() else 0.0, H, W)
                masks_added[label] += 1
                any_mask = True
            frames_with_any += int(any_mask)

            if len(image_paths) >= 16 and (frame_idx + 1) % 16 == 0:
                print(f"  DINOtxt processed {frame_idx + 1}/{len(image_paths)} frames", flush=True)
            del x, patch_tokens, patch_scores
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sparse_tracks = [track for track in tracks.values() if track.get("mask_by_frame")]
    sparse = SparseMaskletOutput(
        tracks=sparse_tracks,
        num_masklets=len(sparse_tracks),
        num_frames=len(image_paths),
        frame_height=H,
        frame_width=W,
        debug={
            "dinotxt_stuff_standalone": {
                "dinov3_root": str(dinov3_root),
                "weights": str(args.weights),
                "backbone_weights": str(args.backbone_weights),
                "bpe_path": str(args.bpe_path),
                "labels": labels,
                "negative_labels": negative_labels,
                "templates": templates,
                "frames": int(len(image_paths)),
                "start_frame": max(int(args.start_frame), 0),
                "input_size": int(args.input_size),
                "score_margin": score_margin,
                "min_area_ratio": min_area,
                "max_area_ratio": max_area,
                "morph_kernel": int(args.morph_kernel),
                "observed_pixels": observed_pixels,
                "masks_added": masks_added,
                "frames_with_any": int(frames_with_any),
                "elapsed_seconds": float(time.time() - t0),
            }
        },
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
        "dinotxt_debug": sparse.debug["dinotxt_stuff_standalone"],
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
