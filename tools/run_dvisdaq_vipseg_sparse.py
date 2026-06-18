#!/usr/bin/env python3
"""Run DVIS-DAQ VIPSeg VPS and save project sparse masklets."""

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
from run_videomt_vspw_sparse import _norm_label, _split_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DVIS-DAQ VIPSeg as sparse video panoptic branch.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dvisdaq_root", default="third_party/DVIS_Plus/DVIS_DAQ")
    parser.add_argument("--config", default="configs/dvis_daq/vipseg/DAQ_Online_R50.yaml")
    parser.add_argument("--checkpoint", default="ckpts/DVIS_DAQ/model_vipseg_online_r50.pth")
    parser.add_argument("--frames_limit", type=int, default=64)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=-1)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--min_area_ratio", type=float, default=0.002)
    parser.add_argument("--max_area_ratio", type=float, default=0.95)
    parser.add_argument("--keep_labels", default="all")
    parser.add_argument("--drop_thing_labels", type=int, default=0)
    parser.add_argument("--drop_stuff_labels", type=int, default=0)
    parser.add_argument("--device_type", default="cuda")
    parser.add_argument("--contact_frames", default="0,8,16,24,32,40,48,56,63")
    return parser.parse_args()


def _load_processing_frames(args: argparse.Namespace) -> Tuple[List[str], List[str], Tuple[int, int]]:
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


def _load_dvisdaq(dvisdaq_root: Path) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    root = dvisdaq_root.resolve()
    demo_root = root / "demo_video"
    for path in (root, demo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.data.detection_utils import read_image
    from detectron2.modeling import build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from mask2former import add_maskformer2_config
    from mask2former_video import add_maskformer2_video_config
    from dvis_Plus import add_ctvis_config, add_dvis_config, add_minvis_config
    from dvis_daq import add_daq_config
    import dvis_Plus.data_video.datasets.builtin  # noqa: F401

    return (
        get_cfg,
        add_deeplab_config,
        add_maskformer2_config,
        add_maskformer2_video_config,
        add_minvis_config,
        add_dvis_config,
        add_ctvis_config,
        add_daq_config,
        MetadataCatalog,
        read_image,
        build_model,
    )


class _DAQVideoPredictor:
    """Minimal inference predictor with non-strict criterion-buffer loading."""

    def __init__(self, cfg: Any):
        import detectron2.data.transforms as T

        self.cfg = cfg.clone()
        self.model = self._build_model()
        self.model.eval()
        weight = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        if "model" in weight:
            weight = weight["model"]
        incompatible = self.model.load_state_dict(weight, strict=False)
        self.load_debug = {
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        }
        if self.load_debug["missing_keys"] or self.load_debug["unexpected_keys"]:
            print(
                "[dvisdaq] non-strict weight load: "
                f"missing={self.load_debug['missing_keys'][:8]} "
                f"unexpected={self.load_debug['unexpected_keys'][:8]}",
                flush=True,
            )
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST,
        )
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def _build_model(self) -> Any:
        from detectron2.modeling import build_model

        return build_model(self.cfg)

    def __call__(self, frames_and_keep: Tuple[List[np.ndarray], bool, int]) -> Dict[str, Any]:
        frames, keep, start_frame_idx = frames_and_keep
        input_frames = []
        height = width = 0
        for original_image in frames:
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            input_frames.append(image)
        inputs = {
            "image": input_frames,
            "height": height,
            "width": width,
            "keep": bool(keep),
            "long_video_start_fidx": int(start_frame_idx),
        }
        with torch.no_grad():
            return self.model([inputs])


def _run_windows(
    predictor: Any,
    read_image: Any,
    image_paths: Sequence[str],
    window_size: int,
    device_type: str,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    frames: List[np.ndarray] = []
    win = int(window_size)
    if win <= 0:
        win = len(image_paths)
    window_start = 0
    for idx, path in enumerate(image_paths):
        frames.append(read_image(str(path), format="BGR"))
        if len(frames) == win or idx == len(image_paths) - 1:
            keep = bool(outputs)
            with torch.no_grad():
                if str(device_type).startswith("cuda") and torch.cuda.is_available():
                    with torch.amp.autocast(device_type="cuda"):
                        predictions = predictor((frames, keep, window_start))
                else:
                    predictions = predictor((frames, keep, window_start))
            outputs.append(predictions)
            window_start += len(frames)
            frames = []
            processed = sum(int(np.asarray(out["pred_masks"]).shape[0]) for out in outputs)
            print(f"  DVIS-DAQ processed {processed}/{len(image_paths)} frames", flush=True)
    return outputs


def _all_categories(metadata: Any) -> Dict[int, Dict[str, Any]]:
    categories = getattr(metadata, "categories", None)
    if not categories:
        raise RuntimeError("DVIS-DAQ metadata does not provide VIPSeg categories")
    return {int(k): dict(v) for k, v in categories.items()}


def _label_allowed(label: str, keep_labels: str) -> bool:
    keep = {_norm_label(x) for x in _split_csv(keep_labels)}
    if not keep or str(keep_labels).strip().lower() == "all":
        return True
    return _norm_label(label) in keep


def _make_sparse(
    outputs: Sequence[Dict[str, Any]],
    categories: Dict[int, Dict[str, Any]],
    height: int,
    width: int,
    keep_labels: str,
    drop_thing_labels: bool,
    drop_stuff_labels: bool,
    min_area_ratio: float,
    max_area_ratio: float,
    debug: Dict[str, Any],
) -> SparseMaskletOutput:
    tracks: Dict[Tuple[str, int], Dict[str, Any]] = {}
    observed_pixels: Dict[str, int] = {}
    skipped_by_area: Dict[str, int] = {}
    segment_records: List[Dict[str, Any]] = []
    global_frame = 0
    for out_index, pred_out in enumerate(outputs):
        pan = pred_out["pred_masks"]
        if hasattr(pan, "detach"):
            pan = pan.detach().cpu()
        pan_np = np.asarray(pan).astype(np.int64)
        if pan_np.ndim != 3:
            raise RuntimeError(f"Unexpected DVIS-DAQ pred_masks shape: {pan_np.shape}")
        segments_infos = list(pred_out.get("segments_infos", []))
        pred_ids = list(pred_out.get("pred_ids", []))
        if len(pred_ids) < len(segments_infos):
            pred_ids = [seg.get("id", idx) for idx, seg in enumerate(segments_infos)]

        for local_t in range(pan_np.shape[0]):
            frame_map = pan_np[local_t]
            if frame_map.shape != (height, width):
                frame_map = cv2.resize(
                    frame_map.astype(np.int32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int64)
            for seg_idx, seg in enumerate(segments_infos):
                seg_id = int(seg.get("id"))
                category_id = int(seg.get("category_id"))
                cat = categories.get(category_id, {"name": f"class_{category_id}", "isthing": 1})
                raw_label = str(cat.get("name", f"class_{category_id}"))
                label = _norm_label(raw_label)
                isthing = bool(int(cat.get("isthing", seg.get("isthing", 1))))
                if drop_thing_labels and isthing:
                    continue
                if drop_stuff_labels and not isthing:
                    continue
                if not _label_allowed(label, keep_labels):
                    continue
                mask = frame_map == seg_id
                pix = int(mask.sum())
                if pix <= 0:
                    continue
                observed_pixels[label] = observed_pixels.get(label, 0) + pix
                area_ratio = float(pix) / float(max(height * width, 1))
                if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
                    skipped_by_area[label] = skipped_by_area.get(label, 0) + 1
                    continue
                pred_id = int(pred_ids[seg_idx]) if seg_idx < len(pred_ids) else seg_id
                source_type = "thing_tracked" if isthing else "stuff_static"
                track_key = (label, pred_id)
                if track_key not in tracks:
                    tracks[track_key] = _make_track(
                        label,
                        source_type,
                        global_frame,
                        height,
                        width,
                        "dvisdaq_vipseg_vps",
                        None,
                    )
                    tracks[track_key]["dvisdaq_pred_id"] = int(pred_id)
                    tracks[track_key]["dvisdaq_category_id"] = int(category_id)
                    tracks[track_key]["dvisdaq_segment_id"] = int(seg_id)
                _write_mask(tracks[track_key], global_frame, mask, 1.0, height, width)
                segment_records.append(
                    {
                        "output_index": int(out_index),
                        "frame": int(global_frame),
                        "label": label,
                        "pred_id": int(pred_id),
                        "segment_id": int(seg_id),
                        "category_id": int(category_id),
                        "isthing": bool(isthing),
                        "area_ratio": float(area_ratio),
                    }
                )
            global_frame += 1

    sparse_tracks = [track for track in tracks.values() if track.get("mask_by_frame")]
    debug.update(
        {
            "categories": {str(k): v for k, v in categories.items()},
            "keep_labels": keep_labels,
            "drop_thing_labels": bool(drop_thing_labels),
            "drop_stuff_labels": bool(drop_stuff_labels),
            "observed_pixels": observed_pixels,
            "skipped_by_area": skipped_by_area,
            "segments_kept": int(len(segment_records)),
            "segment_records_sample": segment_records[:200],
        }
    )
    return SparseMaskletOutput(
        tracks=sparse_tracks,
        num_masklets=len(sparse_tracks),
        num_frames=global_frame,
        frame_height=height,
        frame_width=width,
        debug={"dvisdaq_vipseg_vps_sparse": debug},
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    height, width = proc_shape

    (
        get_cfg,
        add_deeplab_config,
        add_maskformer2_config,
        add_maskformer2_video_config,
        add_minvis_config,
        add_dvis_config,
        add_ctvis_config,
        add_daq_config,
        MetadataCatalog,
        read_image,
        _build_model,
    ) = _load_dvisdaq(Path(args.dvisdaq_root))

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    add_dvis_config(cfg)
    add_ctvis_config(cfg)
    add_daq_config(cfg)
    config_path = str((Path(args.dvisdaq_root) / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config))
    checkpoint_path = str((REPO_ROOT / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint))
    cfg.merge_from_file(config_path)
    cfg.merge_from_list(["MODEL.WEIGHTS", checkpoint_path, "OUTPUT_DIR", str(output_dir)])
    cfg.freeze()
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    categories = _all_categories(metadata)

    t0 = time.time()
    predictor = _DAQVideoPredictor(cfg)
    outputs = _run_windows(predictor, read_image, image_paths, int(args.window_size), str(args.device_type))
    elapsed = time.time() - t0
    debug = {
        "dvisdaq_root": str(Path(args.dvisdaq_root).resolve()),
        "config": config_path,
        "checkpoint": checkpoint_path,
        "input_video": str(args.input_video),
        "frames": int(len(image_paths)),
        "processing_shape_hw": [int(height), int(width)],
        "window_size": int(args.window_size),
        "elapsed_seconds": float(elapsed),
        "model_task": str(cfg.MODEL.MASK_FORMER.TEST.TASK),
        "dataset": list(cfg.DATASETS.TEST),
        "weight_load": getattr(predictor, "load_debug", {}),
    }
    sparse = _make_sparse(
        outputs,
        categories,
        height,
        width,
        str(args.keep_labels),
        bool(int(args.drop_thing_labels)),
        bool(int(args.drop_stuff_labels)),
        float(args.min_area_ratio),
        float(args.max_area_ratio),
        debug,
    )

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_sheet.jpg"
    metrics_path = output_dir / "metrics_summary.json"

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
        "dvisdaq_debug": sparse.debug.get("dvisdaq_vipseg_vps_sparse", {}),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
