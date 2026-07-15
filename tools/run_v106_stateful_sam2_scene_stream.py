#!/usr/bin/env python3
"""Run v106 SAM2 scene streaming without file-level chunk chaining.

This runner is the replacement for the v106 Phase9 file-chain smoke path.  It
keeps one SAM2 video inference state alive for the requested scene segment and
performs frame-by-frame continuation in memory.  Output files are written only
as final artifacts for visual review; they are not used to pass state between
chunks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.audit_v105_baseline_x_sam2_twostage_tracking import (  # noqa: E402
    load_config,
    make_args,
    run as run_v105_baseline,
)
from tools.audit_v105_4dpm_style_per_frame_segmentors import (  # noqa: E402
    parse_frame_ids,
    read_rgb,
    sha256_file,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/v105/baseline_chunk_table/baseline_x_gapadaptive_sam2.generated.yaml"
VARIANT_ID = "v106_stateful_sam2_scene_stream"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _write_side_by_side_video(
    *,
    summary: dict[str, Any],
    output_root: Path,
    variant_id: str,
    fps: float,
) -> dict[str, Any]:
    visual_dir = output_root / "v106_visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)
    records = list(summary.get("records", []))
    if not records:
        raise ValueError("summary contains no records for visual video export")

    frame_paths: list[Path] = []
    for row in records:
        rgb = read_rgb(_resolve(row["rgb_path"]) if "rgb_path" in row else _resolve_frame_path(summary, row))
        overlay_path = _resolve(row["overlay_path"])
        overlay = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
        if overlay is None:
            raise FileNotFoundError(overlay_path)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        if overlay_rgb.shape[:2] != rgb.shape[:2]:
            overlay_rgb = cv2.resize(overlay_rgb, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        panel = np.concatenate([rgb, overlay_rgb], axis=1)
        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        panel_path = visual_dir / f"scene_frame_{int(row['chunk_frame_index']):03d}_id_{int(row['frame_id']):06d}.jpg"
        cv2.imwrite(str(panel_path), panel_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame_paths.append(panel_path)

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise FileNotFoundError(frame_paths[0])
    height, width = first.shape[:2]
    video_path = visual_dir / f"{variant_id}_{summary['scene_id']}_rgb_overlay_{len(frame_paths)}f.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {video_path}")
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    return {
        "schema_version": "stream4d_v106_stateful_visual_video_v1",
        "layout": "RGB frame | v106 stateful SAM2 overlay",
        "path": str(video_path),
        "sha256": sha256_file(video_path),
        "frame_count": int(len(frame_paths)),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "panel_frame_dir": str(visual_dir),
    }


def _resolve_frame_path(summary: dict[str, Any], row: dict[str, Any]) -> Path:
    rgb_root = _resolve(summary.get("rgb_root", "Stream3D/data/scannet/processed"))
    return rgb_root / str(summary["scene_id"]) / "color" / f"{int(row['frame_id'])}.jpg"


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
    parser.add_argument("--stream-prune-invisible-after-frames", type=int, default=6)
    parser.add_argument("--stream-prune-min-visible-area", type=int, default=256)
    parser.add_argument("--stream-empty-cache-every", type=int, default=8)
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=False)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=False)
    parser.add_argument("--gap-max-points", type=int, default=128)
    parser.add_argument("--gap-min-component-area", type=int, default=4096)
    parser.add_argument("--gap-area-per-extra-point", type=int, default=80000)
    parser.add_argument("--gap-max-points-per-component", type=int, default=4)
    parser.add_argument("--gap-min-image-edge-distance-px", type=int, default=0)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--birth-dump-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if str(cli.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    frame_count = int(cli.frame_count)
    if frame_count <= 0:
        frame_count = int(cli.chunk_size) + max(0, int(cli.chunk_count) - 1) * (int(cli.chunk_size) - int(cli.overlap))
    output_root = _resolve(cli.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = _resolve(cli.config)
    config = load_config(config_path)
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
    args.baseline_id = "v106-stateful-sam2-scene-stream"
    args.model_provider = "sam2"
    args.propagation_mode = "streaming_state"
    args.model_dtype = str(cli.model_dtype)
    args.runtime_num_maskmem = int(cli.runtime_num_maskmem)
    args.runtime_max_obj_ptrs_in_encoder = int(cli.runtime_max_obj_ptrs_in_encoder)
    args.runtime_max_cond_frames_in_attn = int(cli.runtime_max_cond_frames_in_attn)
    args.stream_keep_noncond_frames = int(cli.stream_keep_noncond_frames)
    args.stream_prune_invisible_after_frames = int(cli.stream_prune_invisible_after_frames)
    args.stream_prune_min_visible_area = int(cli.stream_prune_min_visible_area)
    args.stream_empty_cache_every = int(cli.stream_empty_cache_every)
    args.offload_video_to_cpu = bool(cli.offload_video_to_cpu)
    args.offload_state_to_cpu = bool(cli.offload_state_to_cpu)
    args.gap_sampler = "component_adaptive"
    args.gap_max_points = int(cli.gap_max_points)
    args.gap_min_component_area = int(cli.gap_min_component_area)
    args.gap_area_per_extra_point = int(cli.gap_area_per_extra_point)
    args.gap_max_points_per_component = int(cli.gap_max_points_per_component)
    args.gap_min_image_edge_distance_px = int(cli.gap_min_image_edge_distance_px)
    args.fps = float(cli.fps)
    variant_id = VARIANT_ID

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    started = time.time()
    run_v105_baseline(args)
    wall_sec = time.time() - started

    run_root = output_root / variant_id
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_version": "stream4d_v106_stateful_sam2_scene_stream_summary_v1",
            "v106_stateful_streaming": {
                "file_level_chunk_chaining_used": False,
                "legacy_phase9_file_chain_used": False,
                "same_process_sam2_predictor_used": True,
                "same_sam2_video_inference_state_used_for_scene": True,
                "model_provider": "sam2",
                "segmentor_provider": str(args.segmentor_name),
                "tracker_provider": str(args.tracker_name),
                "same_video_inference_state_used_for_scene": True,
                "inter_chunk_state_transport": "in_memory_sam2_state_plus_runtime_object_ids",
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
                "stream_empty_cache_every": int(cli.stream_empty_cache_every),
                "offload_video_to_cpu": bool(cli.offload_video_to_cpu),
                "offload_state_to_cpu": bool(cli.offload_state_to_cpu),
                "gap_max_points": int(cli.gap_max_points),
                "gap_min_component_area": int(cli.gap_min_component_area),
            },
            "wrapper_wall_time_sec": float(wall_sec),
            "wrapper_wall_time_human": _format_seconds(wall_sec),
        }
    )
    visual = _write_side_by_side_video(
        summary=summary,
        output_root=run_root,
        variant_id=variant_id,
        fps=float(cli.fps),
    )
    summary["v106_visual_confirmation_video"] = visual
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "video": visual["path"],
                "wall_time_sec": float(wall_sec),
                "file_level_chunk_chaining_used": False,
                "metrics_gate_used_for_decision": False,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


def _format_seconds(seconds: float) -> str:
    total = int(round(float(seconds)))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


if __name__ == "__main__":
    raise SystemExit(main())
