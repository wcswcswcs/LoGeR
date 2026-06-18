#!/usr/bin/env python3
"""Run VidEoMT VSPW VSS and save project sparse masklets.

This keeps the upstream VidEoMT inference path intact, but writes the raw VSS
semantic maps into the LoGeR sparse_masklets_v1 format so they can be fused
with existing SAM/SAM3.1 thing tracks.
"""

from __future__ import annotations

import argparse
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
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import coverage_stats, parse_contact_frames, track_stats  # noqa: E402
from run_video_masklet_front_end import SparseMaskletOutput, collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import _make_track, _write_mask, create_tracking_video_v2, save_sparse_output  # noqa: E402


THING_LABELS = {
    "person",
    "people",
    "man",
    "woman",
    "child",
    "rider",
    "car",
    "bus",
    "truck",
    "train",
    "motorcycle",
    "bicycle",
    "chair",
    "table",
    "desk",
    "lamp",
    "trash_can",
    "trash can",
    "traveling_case_or_trolley_case",
    "case",
    "cabinet",
    "screen",
    "door",
    "window",
    "painting_or_poster",
    "poster",
    "book",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VidEoMT VSPW as sparse semantic branch.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--videomt_root", default="third_party/VidEoMT")
    parser.add_argument("--config", default="configs/VSPW/videomt/dinov2/vit-large/videomt_online_ViTL.yaml")
    parser.add_argument("--checkpoint", default="ckpts/VidEoMT/vspw_vit_large_95.0_64.9.pth")
    parser.add_argument("--frames_limit", type=int, default=64)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--render_video", type=int, default=1)
    parser.add_argument("--render_contact_sheet", type=int, default=1)
    parser.add_argument("--min_area_ratio", type=float, default=0.002)
    parser.add_argument("--max_area_ratio", type=float, default=0.95)
    parser.add_argument("--keep_labels", default="all")
    parser.add_argument("--drop_thing_labels", type=int, default=1)
    parser.add_argument("--device_type", default="cuda")
    parser.add_argument("--contact_frames", default="0,8,16,24,32,40,48,56,63")
    return parser.parse_args()


def _split_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text or "").split(",") if item.strip()]


def _norm_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _load_processing_frames(args: argparse.Namespace) -> tuple[List[str], List[str], tuple[int, int]]:
    image_paths, temp_dir = collect_image_paths(
        args.input_video,
        int(args.start_frame),
        int(args.end_frame),
        int(args.frame_stride),
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
    return list(image_paths), temp_dirs, tuple(int(x) for x in proc_shape)


def _load_videomt(videomt_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    root = videomt_root.resolve()
    vis_root = root / "visualization"
    for path in (root, vis_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data.detection_utils import read_image
    from detectron2.projects.deeplab import add_deeplab_config
    from videomt import add_videomt_config
    from predictor import VisualizationDemo_windows

    return get_cfg, add_deeplab_config, add_videomt_config, MetadataCatalog, read_image, VisualizationDemo_windows


def _metadata_classes(metadata: Any) -> List[str]:
    classes = metadata.get("stuff_classes", None)
    if classes is None:
        classes = metadata.get("thing_classes", None)
    if classes is None:
        raise RuntimeError("VidEoMT metadata does not provide class names")
    return [str(x) for x in classes]


def _selected_class_ids(classes: Sequence[str], keep_labels: str, drop_thing_labels: bool) -> Dict[int, str]:
    keep = {_norm_label(x) for x in _split_csv(keep_labels)}
    use_all = not keep or str(keep_labels).strip().lower() == "all"
    selected: Dict[int, str] = {}
    for idx, label in enumerate(classes):
        norm = _norm_label(label)
        if not use_all and norm not in keep:
            continue
        if drop_thing_labels and norm in {_norm_label(x) for x in THING_LABELS}:
            continue
        selected[int(idx)] = norm
    return selected


def _run_windows(
    demo: Any,
    read_image: Any,
    image_paths: Sequence[str],
    window_size: int,
    device_type: str,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    maps: List[np.ndarray] = []
    confidence_maps: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    win = max(int(window_size), 1)
    if win == -1:
        win = len(image_paths)
    for idx, path in enumerate(image_paths):
        frames.append(read_image(str(path), format="BGR"))
        if len(frames) == win or idx == len(image_paths) - 1:
            keep = bool(maps)
            with torch.no_grad():
                if str(device_type).startswith("cuda") and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda"):
                        predictions, _ = demo.run_on_video(frames, keep=keep)
                else:
                    predictions, _ = demo.run_on_video(frames, keep=keep)
            pred = predictions["pred_masks"]
            confidence = predictions.get("pred_confidence_maps")
            if hasattr(pred, "detach"):
                pred = pred.detach().cpu()
            pred_np = np.asarray(pred).astype(np.int64)
            if pred_np.ndim != 3:
                raise RuntimeError(f"Unexpected VidEoMT pred_masks shape: {pred_np.shape}")
            if confidence is None:
                raise RuntimeError("VidEoMT vss output missing pred_confidence_maps; refusing to fabricate confidence")
            if hasattr(confidence, "detach"):
                confidence = confidence.detach().cpu()
            confidence_np = np.asarray(confidence).astype(np.float32)
            if confidence_np.ndim != 3:
                raise RuntimeError(f"Unexpected VidEoMT pred_confidence_maps shape: {confidence_np.shape}")
            if confidence_np.shape != pred_np.shape:
                raise RuntimeError(
                    f"VidEoMT confidence shape mismatch: masks={pred_np.shape} confidence={confidence_np.shape}"
                )
            maps.extend(pred_np[t] for t in range(pred_np.shape[0]))
            confidence_maps.extend(confidence_np[t] for t in range(confidence_np.shape[0]))
            frames = []
            print(f"  VidEoMT processed {len(maps)}/{len(image_paths)} frames", flush=True)
    return maps, confidence_maps


def _make_sparse(
    pred_maps: Sequence[np.ndarray],
    confidence_maps: Sequence[np.ndarray],
    classes: Sequence[str],
    selected: Dict[int, str],
    height: int,
    width: int,
    min_area_ratio: float,
    max_area_ratio: float,
    debug: Dict[str, Any],
) -> SparseMaskletOutput:
    if len(pred_maps) != len(confidence_maps):
        raise RuntimeError(
            f"VidEoMT output length mismatch: pred_maps={len(pred_maps)} confidence_maps={len(confidence_maps)}"
        )
    tracks = {
        label: _make_track(label, "stuff_static", 0, height, width, "videomt_vspw_vss", None)
        for label in selected.values()
    }
    masks_added = {label: 0 for label in tracks}
    observed_pixels = {str(label): 0 for label in classes}
    skipped_by_area: Dict[str, int] = {label: 0 for label in tracks}
    min_area = float(min_area_ratio)
    max_area = float(max_area_ratio)
    for frame_idx, pred in enumerate(pred_maps):
        if pred.shape != (height, width):
            pred = cv2.resize(pred.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        vals, counts = np.unique(pred, return_counts=True)
        for val, count in zip(vals.tolist(), counts.tolist()):
            if 0 <= int(val) < len(classes):
                observed_pixels[str(classes[int(val)])] += int(count)
        for class_id, label in selected.items():
            mask = pred == int(class_id)
            area_ratio = float(mask.sum()) / float(max(height * width, 1))
            if area_ratio < min_area or area_ratio > max_area:
                skipped_by_area[label] += int(mask.any())
                continue
            _write_mask(tracks[label], frame_idx, mask, 1.0, height, width)
            masks_added[label] += 1
    sparse_tracks = [track for track in tracks.values() if track.get("mask_by_frame")]
    if pred_maps:
        label_maps_np = np.stack([np.asarray(frame, dtype=np.int64) for frame in pred_maps], axis=0)
        confidence_np = np.stack([np.asarray(frame, dtype=np.float32) for frame in confidence_maps], axis=0)
    else:
        label_maps_np = np.zeros((0, height, width), dtype=np.int64)
        confidence_np = np.zeros((0, height, width), dtype=np.float32)
    semantic_segmentation = {
        "format": "videomt_vss_semantic_maps_v1",
        "source": "videomt_vspw_vss",
        "frame_height": int(height),
        "frame_width": int(width),
        "num_frames": int(len(pred_maps)),
        "label_names": list(classes),
        "label_to_id": {str(label): int(idx) for idx, label in enumerate(classes)},
        "label_maps": torch.as_tensor(label_maps_np),
        "confidence_maps": torch.as_tensor(confidence_np, dtype=torch.float32),
        "confidence_source": "pred_confidence_maps",
        "confidence_mode": "max_class_probability",
        "provenance": {
            "input_kind": "videomt_vss",
            "notes": "Dense semantic labels and confidences are taken directly from VidEoMT VSS outputs without any synthetic confidence fallback.",
        },
        "debug": {
            "label_map_shape": list(label_maps_np.shape),
            "confidence_map_shape": list(confidence_np.shape),
            "confidence_min": float(confidence_np.min()) if confidence_np.size else 0.0,
            "confidence_max": float(confidence_np.max()) if confidence_np.size else 0.0,
            "confidence_mean": float(confidence_np.mean()) if confidence_np.size else 0.0,
        },
    }
    debug.update(
        {
            "classes": list(classes),
            "selected_class_ids": {str(k): v for k, v in selected.items()},
            "masks_added": masks_added,
            "skipped_by_area": skipped_by_area,
            "observed_pixels": observed_pixels,
            "semantic_label_shape": list(label_maps_np.shape),
            "semantic_confidence_shape": list(confidence_np.shape),
            "semantic_confidence_min": float(confidence_np.min()) if confidence_np.size else 0.0,
            "semantic_confidence_max": float(confidence_np.max()) if confidence_np.size else 0.0,
            "semantic_confidence_mean": float(confidence_np.mean()) if confidence_np.size else 0.0,
        }
    )
    return SparseMaskletOutput(
        tracks=sparse_tracks,
        num_masklets=len(sparse_tracks),
        num_frames=len(pred_maps),
        frame_height=height,
        frame_width=width,
        debug={"videomt_vspw_vss_sparse": debug},
        semantic_segmentation=semantic_segmentation,
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    height, width = proc_shape

    get_cfg, add_deeplab_config, add_videomt_config, MetadataCatalog, read_image, VisualizationDemo_windows = _load_videomt(Path(args.videomt_root))
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    config_path = str((Path(args.videomt_root) / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    checkpoint_path = str((REPO_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint))
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", checkpoint_path, "OUTPUT_DIR", str(output_dir)])
    cfg.freeze()
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    classes = _metadata_classes(metadata)
    selected = _selected_class_ids(classes, args.keep_labels, bool(int(args.drop_thing_labels)))
    if not selected:
        raise RuntimeError("No VidEoMT classes selected after filtering")

    t0 = time.time()
    demo = VisualizationDemo_windows(cfg)
    pred_maps, confidence_maps = _run_windows(demo, read_image, image_paths, int(args.window_size), str(args.device_type))
    debug = {
        "videomt_root": str(Path(args.videomt_root).resolve()),
        "config": config_path,
        "checkpoint": checkpoint_path,
        "input_video": str(args.input_video),
        "frames": int(len(image_paths)),
        "processing_shape_hw": [int(height), int(width)],
        "window_size": int(args.window_size),
        "drop_thing_labels": bool(int(args.drop_thing_labels)),
        "keep_labels": str(args.keep_labels),
        "min_area_ratio": float(args.min_area_ratio),
        "max_area_ratio": float(args.max_area_ratio),
        "elapsed_seconds": float(time.time() - t0),
    }
    sparse = _make_sparse(
        pred_maps,
        confidence_maps,
        classes,
        selected,
        height,
        width,
        float(args.min_area_ratio),
        float(args.max_area_ratio),
        debug,
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_sheet.jpg"
    metrics_path = output_dir / "metrics_summary.json"

    save_sparse_output(output_pt, sparse)
    if int(args.render_video):
        create_tracking_video_v2(
            image_paths,
            sparse,
            str(output_video),
            fps=int(args.fps),
            mask_alpha=float(args.mask_alpha),
            render_style="clean",
        )
    if int(args.render_contact_sheet):
        _make_single_contact(
            image_paths,
            sparse,
            parse_contact_frames(args.contact_frames, sparse.num_frames),
            contact_path,
            float(args.mask_alpha),
        )
    summary = {
        "output_pt": str(output_pt),
        "output_video": str(output_video) if int(args.render_video) else "",
        "contact_sheet": str(contact_path) if int(args.render_contact_sheet) else "",
        "render_video": bool(int(args.render_video)),
        "render_contact_sheet": bool(int(args.render_contact_sheet)),
        "coverage": coverage_stats(sparse),
        "track_stats": track_stats(sparse),
        "videomt_debug": sparse.debug["videomt_vspw_vss_sparse"],
        "semantic_segmentation": {
            "format": sparse.semantic_segmentation["format"] if sparse.semantic_segmentation else None,
            "shape": list(sparse.semantic_segmentation["label_maps"].shape) if sparse.semantic_segmentation else [],
            "confidence_shape": list(sparse.semantic_segmentation["confidence_maps"].shape) if sparse.semantic_segmentation else [],
            "confidence_min": float(sparse.semantic_segmentation["confidence_maps"].min().item()) if sparse.semantic_segmentation else None,
            "confidence_max": float(sparse.semantic_segmentation["confidence_maps"].max().item()) if sparse.semantic_segmentation else None,
            "confidence_mean": float(sparse.semantic_segmentation["confidence_maps"].float().mean().item()) if sparse.semantic_segmentation else None,
        },
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
