#!/usr/bin/env python3
"""Run v106 SAM2-only rolling video-state streaming.

This variant appends frames into one live SAM2 inference_state instead of
initializing the state from a prebuilt full video directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    PALETTE,
    parse_frame_ids,
    read_rgb,
    sha256_file,
)
from tools.run_v106_stateful_sam2_scene_stream import (  # noqa: E402
    _format_seconds,
    _resolve,
    _write_side_by_side_video,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/v106/v106_stateful_sam2_rolling_scene_stream.yaml"
VARIANT_ID = "v106_stateful_sam2_rolling_scene_stream"
BASELINE_ID = "v106-stateful-sam2-rolling-scene-stream"

_PALETTE_ARRAY = np.asarray(PALETTE, dtype=np.uint8)
_PALETTE_LUT_U16 = np.zeros((65536, 3), dtype=np.uint8)
_PALETTE_LUT_U16[1:] = _PALETTE_ARRAY[(np.arange(1, 65536, dtype=np.int64) - 1) % int(len(_PALETTE_ARRAY))]
_EDGE_KERNEL_2 = np.ones((2, 2), dtype=np.uint8)
_EDGE_KERNEL_3 = np.ones((3, 3), dtype=np.uint8)


def _resolve_frame_path(summary: dict, row: dict) -> Path:
    rgb_root = _resolve(summary.get("rgb_root", "Stream3D/data/scannet/processed"))
    return rgb_root / str(summary["scene_id"]) / "color" / f"{int(row['frame_id'])}.jpg"


def _overlay_label_fast(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    h, w = rgb.shape[:2]
    if label.shape[:2] != (h, w):
        label = cv2.resize(label.astype(np.uint16, copy=False), (w, h), interpolation=cv2.INTER_NEAREST)
    label_u16 = label.astype(np.uint16, copy=False)
    fg = label_u16 > 0
    if not bool(np.any(fg)):
        return rgb.copy()

    color = _PALETTE_LUT_U16[label_u16]
    blended = cv2.addWeighted(rgb, 1.0 - float(alpha), color, float(alpha), 0.0)
    blended[~fg] = rgb[~fg]

    edge = cv2.morphologyEx(label_u16, cv2.MORPH_GRADIENT, _EDGE_KERNEL_3)
    edge_u8 = cv2.dilate(((edge > 0) & fg).astype(np.uint8), _EDGE_KERNEL_2, iterations=1)
    blended[edge_u8 > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return blended


def _write_side_by_side_video_from_labels(
    *,
    summary: dict,
    output_root: Path,
    variant_id: str,
    fps: float,
    compact_video: bool = False,
) -> dict:
    visual_dir = output_root / "v106_visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)
    records = list(summary.get("records", []))
    if not records:
        raise ValueError("summary contains no records for visual video export")

    key_indices = {0, len(records) // 2, len(records) - 1, min(33, len(records) - 1)}
    saved_panel_paths: list[Path] = []
    writer: cv2.VideoWriter | None = None
    width = 0
    height = 0
    video_path = visual_dir / f"{variant_id}_{summary['scene_id']}_rgb_overlay_{len(records)}f.mp4"
    try:
        for row_index, row in enumerate(records):
            rgb = read_rgb(_resolve(row["rgb_path"]) if "rgb_path" in row else _resolve_frame_path(summary, row))
            label_path = _resolve(row["label_path"])
            label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
            if label is None:
                raise FileNotFoundError(label_path)
            if label.ndim == 3:
                label = label[:, :, 0]
            if label.shape[:2] != rgb.shape[:2]:
                label = cv2.resize(label, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay_rgb = _overlay_label_fast(rgb, label)
            frame_rgb = overlay_rgb if compact_video else np.concatenate([rgb, overlay_rgb], axis=1)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"failed to open VideoWriter for {video_path}")
            if frame_bgr.shape[:2] != (height, width):
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame_bgr)
            if row_index in key_indices:
                panel = np.concatenate([rgb, overlay_rgb], axis=1)
                panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
                panel_path = visual_dir / f"scene_frame_{int(row['chunk_frame_index']):03d}_id_{int(row['frame_id']):06d}.jpg"
                cv2.imwrite(str(panel_path), panel_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                saved_panel_paths.append(panel_path)
    finally:
        if writer is not None:
            writer.release()
    if writer is None:
        raise RuntimeError("failed to create visual video writer")
    return {
        "schema_version": "stream4d_v106_stateful_visual_video_v1",
        "layout": (
            "v106 SAM2 rolling-state overlay from labels; key panels are RGB frame | overlay"
            if compact_video
            else "RGB frame | v106 SAM2 rolling-state overlay from labels"
        ),
        "path": str(video_path),
        "sha256": sha256_file(video_path),
        "frame_count": int(len(records)),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "panel_frame_dir": str(visual_dir),
        "source": "labels",
        "compact_video": bool(compact_video),
        "saved_panel_frame_count": int(len(saved_panel_paths)),
        "saved_panel_frame_paths": [str(path) for path in saved_panel_paths],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rgb-root", default=None)
    parser.add_argument("--frame-start", type=int, required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--frame-ids", default="")
    parser.add_argument("--chunk-count", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--overlap", type=int, default=3)
    parser.add_argument("--seed", type=int, default=105)
    parser.add_argument("--gpu", default="")
    parser.add_argument("--model-dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--runtime-num-maskmem", type=int, default=3)
    parser.add_argument("--runtime-max-obj-ptrs-in-encoder", type=int, default=8)
    parser.add_argument("--runtime-max-cond-frames-in-attn", type=int, default=4)
    parser.add_argument("--stream-keep-noncond-frames", type=int, default=8)
    parser.add_argument("--stream-prune-invisible-after-frames", type=int, default=18)
    parser.add_argument("--stream-prune-min-visible-area", type=int, default=256)
    parser.add_argument("--stream-prune-max-visible-area", type=int, default=None)
    parser.add_argument("--stream-prune-max-visible-area-ratio", type=float, default=None)
    parser.add_argument("--stream-empty-cache-every", type=int, default=None)
    parser.add_argument("--stream-empty-cache-on-prune", dest="stream_empty_cache_on_prune", action="store_true", default=None)
    parser.add_argument("--no-stream-empty-cache-on-prune", dest="stream_empty_cache_on_prune", action="store_false")
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=False)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    parser.add_argument("--gap-max-points", type=int, default=64)
    parser.add_argument("--gap-min-component-area", type=int, default=4096)
    parser.add_argument("--gap-area-per-extra-point", type=int, default=80000)
    parser.add_argument("--gap-max-points-per-component", type=int, default=3)
    parser.add_argument("--gap-min-image-edge-distance-px", type=int, default=0)
    parser.add_argument("--disable-gap-birth", action="store_true", default=False)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--birth-dump-dir", default="")
    parser.add_argument("--skip-visual-export", action="store_true", default=False)
    parser.add_argument("--lean-visual-export", action="store_true", default=False)
    parser.add_argument("--label-only-visual-export", action="store_true", default=False)
    parser.add_argument("--compact-visual-video", action="store_true", default=False)
    parser.add_argument("--birth-admission-min-area", type=int, default=None)
    parser.add_argument("--birth-admission-max-area", type=int, default=None)
    parser.add_argument("--birth-admission-every", type=int, default=None)
    parser.add_argument("--birth-admission-max-per-frame", type=int, default=None)
    parser.add_argument("--birth-admission-persistence-iou", type=float, default=None)
    parser.add_argument("--birth-admission-persistence-hits", type=int, default=None)
    parser.add_argument("--birth-admission-pending-ttl", type=int, default=None)
    parser.add_argument("--birth-admission-persistence-min-area", type=int, default=None)
    parser.add_argument("--birth-admission-persistence-max-per-frame", type=int, default=None)
    parser.add_argument("--birth-admission-immediate-area", type=int, default=None)
    parser.add_argument("--birth-admission-rescue-min-visible-count", type=int, default=None)
    parser.add_argument("--birth-admission-rescue-min-foreground-ratio", type=float, default=None)
    parser.add_argument("--birth-admission-appearance-enabled", action="store_true", default=False)
    parser.add_argument("--birth-admission-appearance-min-iou", type=float, default=None)
    parser.add_argument("--birth-admission-appearance-max-color-distance", type=float, default=None)
    parser.add_argument("--birth-admission-appearance-max-centroid-distance", type=float, default=None)
    parser.add_argument("--birth-admission-appearance-max-area-ratio", type=float, default=None)
    parser.add_argument("--birth-transaction-enabled", action="store_true", default=False)
    parser.add_argument("--birth-transaction-min-pending", type=int, default=None)
    parser.add_argument("--birth-transaction-max-delay-frames", type=int, default=None)
    parser.add_argument("--birth-transaction-immediate-area", type=int, default=None)
    parser.add_argument("--birth-transaction-min-total-area", type=int, default=None)
    parser.add_argument("--birth-recon-prune-keep-frames", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if str(cli.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from tools.audit_v106_sam2_rolling_state import (  # noqa: PLC0415
        get_rolling_stats,
        load_config,
        make_args,
        run as run_rolling,
    )

    frame_count = int(cli.frame_count)
    if frame_count <= 0:
        frame_count = int(cli.chunk_size) + max(0, int(cli.chunk_count) - 1) * (int(cli.chunk_size) - int(cli.overlap))
    output_root = _resolve(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = _resolve(cli.config)
    config = load_config(config_path)
    opt_cfg = config.get("optimization", {})
    sam2_cfg = config.get("sam2", {})
    skip_visual_export = bool(cli.skip_visual_export or opt_cfg.get("skip_visual_export", False))
    lean_visual_export = bool(cli.lean_visual_export or opt_cfg.get("lean_visual_export", False))
    label_only_visual_export = bool(cli.label_only_visual_export or opt_cfg.get("label_only_visual_export", False))
    compact_visual_video = bool(cli.compact_visual_video or opt_cfg.get("compact_visual_video", False))
    stream_empty_cache_on_prune = bool(
        cli.stream_empty_cache_on_prune
        if cli.stream_empty_cache_on_prune is not None
        else opt_cfg.get("stream_empty_cache_on_prune", True)
    )
    stream_empty_cache_every = int(
        cli.stream_empty_cache_every
        if cli.stream_empty_cache_every is not None
        else opt_cfg.get("stream_empty_cache_every", sam2_cfg.get("stream_empty_cache_every", 8))
    )
    birth_admission_min_area = int(
        cli.birth_admission_min_area
        if cli.birth_admission_min_area is not None
        else opt_cfg.get("birth_admission_min_area", 0)
    )
    birth_admission_max_area = int(
        cli.birth_admission_max_area
        if cli.birth_admission_max_area is not None
        else opt_cfg.get("birth_admission_max_area", 0)
    )
    birth_admission_every = int(
        cli.birth_admission_every
        if cli.birth_admission_every is not None
        else opt_cfg.get("birth_admission_every", 1)
    )
    birth_admission_max_per_frame = int(
        cli.birth_admission_max_per_frame
        if cli.birth_admission_max_per_frame is not None
        else opt_cfg.get("birth_admission_max_per_frame", 0)
    )
    birth_admission_persistence_iou = float(
        cli.birth_admission_persistence_iou
        if cli.birth_admission_persistence_iou is not None
        else opt_cfg.get("birth_admission_persistence_iou", 0.0)
    )
    birth_admission_persistence_hits = int(
        cli.birth_admission_persistence_hits
        if cli.birth_admission_persistence_hits is not None
        else opt_cfg.get("birth_admission_persistence_hits", 0)
    )
    birth_admission_pending_ttl = int(
        cli.birth_admission_pending_ttl
        if cli.birth_admission_pending_ttl is not None
        else opt_cfg.get("birth_admission_pending_ttl", 0)
    )
    birth_admission_persistence_min_area = int(
        cli.birth_admission_persistence_min_area
        if cli.birth_admission_persistence_min_area is not None
        else opt_cfg.get("birth_admission_persistence_min_area", 0)
    )
    birth_admission_persistence_max_per_frame = int(
        cli.birth_admission_persistence_max_per_frame
        if cli.birth_admission_persistence_max_per_frame is not None
        else opt_cfg.get("birth_admission_persistence_max_per_frame", 0)
    )
    birth_admission_immediate_area = int(
        cli.birth_admission_immediate_area
        if cli.birth_admission_immediate_area is not None
        else opt_cfg.get("birth_admission_immediate_area", 0)
    )
    birth_admission_rescue_min_visible_count = int(
        cli.birth_admission_rescue_min_visible_count
        if cli.birth_admission_rescue_min_visible_count is not None
        else opt_cfg.get("birth_admission_rescue_min_visible_count", 0)
    )
    birth_admission_rescue_min_foreground_ratio = float(
        cli.birth_admission_rescue_min_foreground_ratio
        if cli.birth_admission_rescue_min_foreground_ratio is not None
        else opt_cfg.get("birth_admission_rescue_min_foreground_ratio", 0.0)
    )
    birth_admission_appearance_enabled = bool(
        cli.birth_admission_appearance_enabled or opt_cfg.get("birth_admission_appearance_enabled", False)
    )
    birth_admission_appearance_min_iou = float(
        cli.birth_admission_appearance_min_iou
        if cli.birth_admission_appearance_min_iou is not None
        else opt_cfg.get("birth_admission_appearance_min_iou", 0.02)
    )
    birth_admission_appearance_max_color_distance = float(
        cli.birth_admission_appearance_max_color_distance
        if cli.birth_admission_appearance_max_color_distance is not None
        else opt_cfg.get("birth_admission_appearance_max_color_distance", 0.16)
    )
    birth_admission_appearance_max_centroid_distance = float(
        cli.birth_admission_appearance_max_centroid_distance
        if cli.birth_admission_appearance_max_centroid_distance is not None
        else opt_cfg.get("birth_admission_appearance_max_centroid_distance", 96.0)
    )
    birth_admission_appearance_max_area_ratio = float(
        cli.birth_admission_appearance_max_area_ratio
        if cli.birth_admission_appearance_max_area_ratio is not None
        else opt_cfg.get("birth_admission_appearance_max_area_ratio", 4.0)
    )
    birth_transaction_enabled = bool(
        cli.birth_transaction_enabled or opt_cfg.get("birth_transaction_enabled", False)
    )
    birth_transaction_min_pending = int(
        cli.birth_transaction_min_pending
        if cli.birth_transaction_min_pending is not None
        else opt_cfg.get("birth_transaction_min_pending", 0)
    )
    birth_transaction_max_delay_frames = int(
        cli.birth_transaction_max_delay_frames
        if cli.birth_transaction_max_delay_frames is not None
        else opt_cfg.get("birth_transaction_max_delay_frames", 0)
    )
    birth_transaction_immediate_area = int(
        cli.birth_transaction_immediate_area
        if cli.birth_transaction_immediate_area is not None
        else opt_cfg.get("birth_transaction_immediate_area", 0)
    )
    birth_transaction_min_total_area = int(
        cli.birth_transaction_min_total_area
        if cli.birth_transaction_min_total_area is not None
        else opt_cfg.get("birth_transaction_min_total_area", 0)
    )
    birth_recon_prune_keep_frames = int(
        cli.birth_recon_prune_keep_frames
        if cli.birth_recon_prune_keep_frames is not None
        else opt_cfg.get("birth_recon_prune_keep_frames", 0)
    )
    stream_prune_max_visible_area = int(
        cli.stream_prune_max_visible_area
        if cli.stream_prune_max_visible_area is not None
        else opt_cfg.get("stream_prune_max_visible_area", 0)
    )
    stream_prune_max_visible_area_ratio = float(
        cli.stream_prune_max_visible_area_ratio
        if cli.stream_prune_max_visible_area_ratio is not None
        else opt_cfg.get("stream_prune_max_visible_area_ratio", 0.0)
    )
    baseline_cli = SimpleNamespace(
        config=str(config_path),
        scene_id=str(cli.scene_id),
        rgb_root=cli.rgb_root,
        frame_start=int(cli.frame_start),
        frame_stride=int(cli.frame_stride),
        frame_count=int(frame_count),
        frame_ids=str(cli.frame_ids or ""),
        output_root=str(output_root),
        seed=int(cli.seed),
        birth_dump_dir=str(cli.birth_dump_dir or ""),
    )
    args = make_args(config, baseline_cli)
    args.output_root = str(output_root)
    args.variant_id = VARIANT_ID
    args.baseline_id = BASELINE_ID
    args.model_provider = "sam2"
    args.propagation_mode = "streaming_state"
    args.model_dtype = str(cli.model_dtype)
    args.runtime_num_maskmem = int(cli.runtime_num_maskmem)
    args.runtime_max_obj_ptrs_in_encoder = int(cli.runtime_max_obj_ptrs_in_encoder)
    args.runtime_max_cond_frames_in_attn = int(cli.runtime_max_cond_frames_in_attn)
    args.stream_keep_noncond_frames = int(cli.stream_keep_noncond_frames)
    args.stream_prune_invisible_after_frames = int(cli.stream_prune_invisible_after_frames)
    args.stream_prune_min_visible_area = int(cli.stream_prune_min_visible_area)
    args.stream_prune_max_visible_area = int(stream_prune_max_visible_area)
    args.stream_prune_max_visible_area_ratio = float(stream_prune_max_visible_area_ratio)
    args.stream_empty_cache_every = int(stream_empty_cache_every)
    args.stream_empty_cache_on_prune = bool(stream_empty_cache_on_prune)
    args.offload_video_to_cpu = bool(cli.offload_video_to_cpu)
    args.offload_state_to_cpu = bool(cli.offload_state_to_cpu)
    args.gap_sampler = "component_adaptive"
    args.gap_max_points = int(cli.gap_max_points)
    args.gap_min_component_area = int(cli.gap_min_component_area)
    args.gap_area_per_extra_point = int(cli.gap_area_per_extra_point)
    args.gap_max_points_per_component = int(cli.gap_max_points_per_component)
    args.gap_min_image_edge_distance_px = int(cli.gap_min_image_edge_distance_px)
    args.disable_gap_birth = bool(cli.disable_gap_birth)
    args.fps = float(cli.fps)
    args.skip_visual_export = bool(skip_visual_export)
    args.lean_visual_export = bool(lean_visual_export)
    args.label_only_visual_export = bool(label_only_visual_export)
    args.compact_visual_video = bool(compact_visual_video)
    args.birth_admission_min_area = int(birth_admission_min_area)
    args.birth_admission_max_area = int(birth_admission_max_area)
    args.birth_admission_every = int(birth_admission_every)
    args.birth_admission_max_per_frame = int(birth_admission_max_per_frame)
    args.birth_admission_persistence_iou = float(birth_admission_persistence_iou)
    args.birth_admission_persistence_hits = int(birth_admission_persistence_hits)
    args.birth_admission_pending_ttl = int(birth_admission_pending_ttl)
    args.birth_admission_persistence_min_area = int(birth_admission_persistence_min_area)
    args.birth_admission_persistence_max_per_frame = int(birth_admission_persistence_max_per_frame)
    args.birth_admission_immediate_area = int(birth_admission_immediate_area)
    args.birth_admission_rescue_min_visible_count = int(birth_admission_rescue_min_visible_count)
    args.birth_admission_rescue_min_foreground_ratio = float(birth_admission_rescue_min_foreground_ratio)
    args.birth_admission_appearance_enabled = bool(birth_admission_appearance_enabled)
    args.birth_admission_appearance_min_iou = float(birth_admission_appearance_min_iou)
    args.birth_admission_appearance_max_color_distance = float(birth_admission_appearance_max_color_distance)
    args.birth_admission_appearance_max_centroid_distance = float(birth_admission_appearance_max_centroid_distance)
    args.birth_admission_appearance_max_area_ratio = float(birth_admission_appearance_max_area_ratio)
    args.birth_transaction_enabled = bool(birth_transaction_enabled)
    args.birth_transaction_min_pending = int(birth_transaction_min_pending)
    args.birth_transaction_max_delay_frames = int(birth_transaction_max_delay_frames)
    args.birth_transaction_immediate_area = int(birth_transaction_immediate_area)
    args.birth_transaction_min_total_area = int(birth_transaction_min_total_area)
    args.birth_recon_prune_keep_frames = int(birth_recon_prune_keep_frames)

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    started = time.time()
    run_rolling(args)
    wall_sec = time.time() - started

    run_root = output_root / VARIANT_ID
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rolling_stats = get_rolling_stats()
    summary.update(
        {
            "schema_version": "stream4d_v106_stateful_sam2_rolling_scene_stream_summary_v1",
            "v106_stateful_streaming": {
                "file_level_chunk_chaining_used": False,
                "legacy_phase9_file_chain_used": False,
                "same_process_sam2_predictor_used": True,
                "same_sam2_video_inference_state_used_for_scene": True,
                "model_provider": "sam2",
                "segmentor_provider": str(args.segmentor_name),
                "tracker_provider": str(args.tracker_name),
                "state_init_mode": "empty_streaming_state_then_append_frames",
                "rolling_frame_append_used": True,
                "full_video_frame_preload_used": False,
                "rolling_image_cache_prune_used": True,
                "inter_chunk_state_transport": "rolling_sam2_inference_state_memory_cache_objptrs",
                "metrics_gate_used_for_decision": False,
                "visual_confirmation_required": True,
                "chunk_count": int(cli.chunk_count),
                "chunk_size": int(cli.chunk_size),
                "overlap": int(cli.overlap),
                "raw_chunk_frames": int(cli.chunk_count) * int(cli.chunk_size),
                "unique_scene_frames": int(len(frame_ids)),
                "frame_ids": [int(v) for v in frame_ids],
                "model_dtype": str(cli.model_dtype),
                "runtime_num_maskmem": int(cli.runtime_num_maskmem),
                "runtime_max_obj_ptrs_in_encoder": int(cli.runtime_max_obj_ptrs_in_encoder),
                "runtime_max_cond_frames_in_attn": int(cli.runtime_max_cond_frames_in_attn),
                "stream_keep_noncond_frames": int(cli.stream_keep_noncond_frames),
                "stream_prune_invisible_after_frames": int(cli.stream_prune_invisible_after_frames),
                "stream_prune_min_visible_area": int(cli.stream_prune_min_visible_area),
                "stream_prune_max_visible_area": int(stream_prune_max_visible_area),
                "stream_prune_max_visible_area_ratio": float(stream_prune_max_visible_area_ratio),
                "compact_visual_video": bool(compact_visual_video),
                "stream_empty_cache_every": int(stream_empty_cache_every),
                "stream_empty_cache_on_prune": bool(stream_empty_cache_on_prune),
                "offload_video_to_cpu": bool(cli.offload_video_to_cpu),
                "offload_state_to_cpu": bool(cli.offload_state_to_cpu),
                "gap_max_points": int(cli.gap_max_points),
                "gap_min_component_area": int(cli.gap_min_component_area),
                "disable_gap_birth": bool(cli.disable_gap_birth),
            },
            "v106_runtime_optimization": {
                "skip_visual_export": bool(skip_visual_export),
                "lean_visual_export": bool(lean_visual_export),
                "label_only_visual_export": bool(label_only_visual_export),
                "compact_visual_video": bool(compact_visual_video),
                "birth_admission_min_area": int(birth_admission_min_area),
                "birth_admission_max_area": int(birth_admission_max_area),
                "birth_admission_every": int(birth_admission_every),
                "birth_admission_max_per_frame": int(birth_admission_max_per_frame),
                "birth_admission_persistence_iou": float(birth_admission_persistence_iou),
                "birth_admission_persistence_hits": int(birth_admission_persistence_hits),
                "birth_admission_pending_ttl": int(birth_admission_pending_ttl),
                "birth_admission_persistence_min_area": int(birth_admission_persistence_min_area),
                "birth_admission_persistence_max_per_frame": int(birth_admission_persistence_max_per_frame),
                "birth_admission_immediate_area": int(birth_admission_immediate_area),
                "birth_admission_rescue_min_visible_count": int(birth_admission_rescue_min_visible_count),
                "birth_admission_rescue_min_foreground_ratio": float(birth_admission_rescue_min_foreground_ratio),
                "birth_admission_appearance_enabled": bool(birth_admission_appearance_enabled),
                "birth_admission_appearance_min_iou": float(birth_admission_appearance_min_iou),
                "birth_admission_appearance_max_color_distance": float(
                    birth_admission_appearance_max_color_distance
                ),
                "birth_admission_appearance_max_centroid_distance": float(
                    birth_admission_appearance_max_centroid_distance
                ),
                "birth_admission_appearance_max_area_ratio": float(birth_admission_appearance_max_area_ratio),
                "birth_transaction_enabled": bool(birth_transaction_enabled),
                "birth_transaction_min_pending": int(birth_transaction_min_pending),
                "birth_transaction_max_delay_frames": int(birth_transaction_max_delay_frames),
                "birth_transaction_immediate_area": int(birth_transaction_immediate_area),
                "birth_transaction_min_total_area": int(birth_transaction_min_total_area),
                "birth_recon_prune_keep_frames": int(birth_recon_prune_keep_frames),
                "stream_prune_max_visible_area": int(stream_prune_max_visible_area),
                "stream_prune_max_visible_area_ratio": float(stream_prune_max_visible_area_ratio),
                "stream_empty_cache_every": int(stream_empty_cache_every),
                "stream_empty_cache_on_prune": bool(stream_empty_cache_on_prune),
            },
            "v106_sam2_rolling_state": rolling_stats,
            "wrapper_wall_time_sec": float(wall_sec),
            "wrapper_wall_time_human": _format_seconds(wall_sec),
        }
    )
    visual_started = time.time()
    if skip_visual_export:
        visual = {
            "schema_version": "stream4d_v106_stateful_visual_video_v1",
            "layout": "RGB frame | v106 SAM2 rolling-state overlay",
            "path": "",
            "sha256": "",
            "frame_count": 0,
            "fps": float(cli.fps),
            "width": 0,
            "height": 0,
            "panel_frame_dir": "",
            "skipped": True,
            "reason": "skip_visual_export",
        }
    elif label_only_visual_export:
        visual = _write_side_by_side_video_from_labels(
            summary=summary,
            output_root=run_root,
            variant_id=VARIANT_ID,
            fps=float(cli.fps),
            compact_video=bool(compact_visual_video),
        )
        if not compact_visual_video:
            visual["layout"] = "RGB frame | v106 SAM2 rolling-state overlay from labels"
        visual["skipped"] = False
    else:
        visual = _write_side_by_side_video(
            summary=summary,
            output_root=run_root,
            variant_id=VARIANT_ID,
            fps=float(cli.fps),
        )
        visual["layout"] = "RGB frame | v106 SAM2 rolling-state overlay"
        visual["skipped"] = False
    visual_export_sec = float(time.time() - visual_started)
    summary["v106_visual_confirmation_video"] = visual
    summary["v106_visual_export_runtime_sec"] = visual_export_sec
    summary["wrapper_total_with_v106_visual_export_wall_time_sec"] = float(time.time() - started)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "video": visual["path"],
                "wall_time_sec": float(wall_sec),
                "visual_export_sec": visual_export_sec,
                "rolling_frame_append_used": True,
                "full_video_frame_preload_used": False,
                "metrics_gate_used_for_decision": False,
                "skip_visual_export": bool(skip_visual_export),
                "lean_visual_export": bool(lean_visual_export),
                "label_only_visual_export": bool(label_only_visual_export),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
