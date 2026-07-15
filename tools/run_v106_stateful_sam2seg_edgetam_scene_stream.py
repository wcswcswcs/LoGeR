#!/usr/bin/env python3
"""Run v106 stateful scene streaming with SAM2 segmentation and EdgeTAM tracking."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
STREAM3D_ROOT = REPO_ROOT / "Stream3D"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STREAM3D_ROOT) not in sys.path:
    sys.path.insert(1, str(STREAM3D_ROOT))

from tools.run_v106_stateful_edgetam_scene_stream import (  # noqa: E402
    _aggregate_frame_diagnostics,
    _format_seconds,
    _resolve,
    _write_side_by_side_video,
    parse_frame_ids,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/v106/v106_stateful_sam2seg_edgetam_scene_stream.yaml"
VARIANT_ID = "v106_stateful_sam2seg_edgetam_tracker"
BASELINE_ID = "v106-stateful-sam2seg-edgetam-tracker"


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
    parser.add_argument("--model-dtype", default=None, choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--edgetam-root", default="")
    parser.add_argument("--edgetam-checkpoint", default="")
    parser.add_argument("--edgetam-model-cfg", default="")
    parser.add_argument("--runtime-num-maskmem", type=int, default=None)
    parser.add_argument("--runtime-max-obj-ptrs-in-encoder", type=int, default=None)
    parser.add_argument("--runtime-max-cond-frames-in-attn", type=int, default=None)
    parser.add_argument("--stream-keep-noncond-frames", type=int, default=None)
    parser.add_argument("--stream-prune-invisible-after-frames", type=int, default=None)
    parser.add_argument("--stream-prune-min-visible-area", type=int, default=None)
    parser.add_argument("--stream-empty-cache-every", type=int, default=None)
    parser.add_argument("--offload-video-to-cpu", action="store_true", default=None)
    parser.add_argument("--offload-state-to-cpu", action="store_true", default=None)
    parser.add_argument("--gap-max-points", type=int, default=None)
    parser.add_argument("--gap-min-component-area", type=int, default=None)
    parser.add_argument("--gap-area-per-extra-point", type=int, default=None)
    parser.add_argument("--gap-max-points-per-component", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--birth-dump-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if str(cli.gpu).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu).strip()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from tools.audit_v106_sam2seg_edgetam_tracking import (  # noqa: PLC0415
        get_runtime_stats,
        load_config,
        make_args,
        run as run_hybrid,
    )

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
    args.baseline_id = BASELINE_ID
    args.model_provider = "sam2seg_edgetam_tracker"
    args.segmentor_name = "sam2.1_hiera_large"
    args.tracker_name = "edgetam"
    args.tracker_backend = "edgetam"
    args.tracker = "edgetam"
    if str(cli.edgetam_root).strip():
        args.edgetam_root = str(cli.edgetam_root)
    if str(cli.edgetam_checkpoint).strip():
        args.edgetam_checkpoint = str(cli.edgetam_checkpoint)
    if str(cli.edgetam_model_cfg).strip():
        args.edgetam_model_cfg = str(cli.edgetam_model_cfg)
    args.propagation_mode = "streaming_state"
    if cli.model_dtype is not None:
        args.model_dtype = str(cli.model_dtype)
    if cli.runtime_num_maskmem is not None:
        args.runtime_num_maskmem = int(cli.runtime_num_maskmem)
    if cli.runtime_max_obj_ptrs_in_encoder is not None:
        args.runtime_max_obj_ptrs_in_encoder = int(cli.runtime_max_obj_ptrs_in_encoder)
    if cli.runtime_max_cond_frames_in_attn is not None:
        args.runtime_max_cond_frames_in_attn = int(cli.runtime_max_cond_frames_in_attn)
    if cli.stream_keep_noncond_frames is not None:
        args.stream_keep_noncond_frames = int(cli.stream_keep_noncond_frames)
    if cli.stream_prune_invisible_after_frames is not None:
        args.stream_prune_invisible_after_frames = int(cli.stream_prune_invisible_after_frames)
    if cli.stream_prune_min_visible_area is not None:
        args.stream_prune_min_visible_area = int(cli.stream_prune_min_visible_area)
    if cli.stream_empty_cache_every is not None:
        args.stream_empty_cache_every = int(cli.stream_empty_cache_every)
    if cli.offload_video_to_cpu is not None:
        args.offload_video_to_cpu = bool(cli.offload_video_to_cpu)
    if cli.offload_state_to_cpu is not None:
        args.offload_state_to_cpu = bool(cli.offload_state_to_cpu)
    args.gap_sampler = "component_adaptive"
    if cli.gap_max_points is not None:
        args.gap_max_points = int(cli.gap_max_points)
    if cli.gap_min_component_area is not None:
        args.gap_min_component_area = int(cli.gap_min_component_area)
    if cli.gap_area_per_extra_point is not None:
        args.gap_area_per_extra_point = int(cli.gap_area_per_extra_point)
    if cli.gap_max_points_per_component is not None:
        args.gap_max_points_per_component = int(cli.gap_max_points_per_component)
    if cli.fps is not None:
        args.fps = float(cli.fps)

    frame_ids = parse_frame_ids(str(args.frame_ids), int(args.frame_start), int(args.frame_stride), int(args.frame_count))
    started = time.time()
    run_hybrid(args)
    wall_sec = time.time() - started

    run_root = output_root / VARIANT_ID
    summary_path = run_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_version": "stream4d_v106_stateful_sam2seg_edgetam_scene_stream_summary_v1",
            "v106_stateful_streaming": {
                "file_level_chunk_chaining_used": False,
                "legacy_phase9_file_chain_used": False,
                "same_process_hybrid_predictors_used": True,
                "same_edgetam_video_inference_state_used_for_scene": True,
                "model_provider": "sam2seg_edgetam_tracker",
                "segmentor_provider": "sam2",
                "tracker_provider": "edgetam",
                "inter_chunk_state_transport": "in_memory_edgetam_state_plus_runtime_object_ids",
                "metrics_gate_used_for_decision": False,
                "visual_confirmation_required": True,
                "chunk_count": int(cli.chunk_count),
                "chunk_size": int(cli.chunk_size),
                "overlap": int(cli.overlap),
                "raw_chunk_frames": int(cli.chunk_count) * int(cli.chunk_size),
                "unique_scene_frames": int(len(frame_ids)),
                "frame_ids": [int(v) for v in frame_ids],
                "model_dtype": str(args.model_dtype),
                "sam2_checkpoint": str(args.sam2_checkpoint),
                "sam2_model_cfg": str(args.sam2_model_cfg),
                "edgetam_root": str(args.edgetam_root),
                "edgetam_checkpoint": str(args.edgetam_checkpoint),
                "edgetam_model_cfg": str(args.edgetam_model_cfg),
                "runtime_num_maskmem": int(args.runtime_num_maskmem),
                "runtime_max_obj_ptrs_in_encoder": int(args.runtime_max_obj_ptrs_in_encoder),
                "runtime_max_cond_frames_in_attn": int(args.runtime_max_cond_frames_in_attn),
                "stream_keep_noncond_frames": int(args.stream_keep_noncond_frames),
                "stream_prune_invisible_after_frames": int(args.stream_prune_invisible_after_frames),
                "stream_prune_min_visible_area": int(args.stream_prune_min_visible_area),
                "stream_empty_cache_every": int(args.stream_empty_cache_every),
                "offload_video_to_cpu": bool(cli.offload_video_to_cpu),
                "offload_state_to_cpu": bool(cli.offload_state_to_cpu),
                "gap_max_points": int(args.gap_max_points),
                "gap_min_component_area": int(args.gap_min_component_area),
                "gap_area_per_extra_point": int(args.gap_area_per_extra_point),
                "gap_max_points_per_component": int(args.gap_max_points_per_component),
                "edgetam_memory_admission_policy": str(getattr(args, "edgetam_memory_admission_policy", "all")),
                "edgetam_memory_admit_min_area": int(getattr(args, "edgetam_memory_admit_min_area", 0)),
                "edgetam_memory_admit_interval": int(getattr(args, "edgetam_memory_admit_interval", 1)),
                "edgetam_memory_admit_warmup_frames": int(getattr(args, "edgetam_memory_admit_warmup_frames", 0)),
                "edgetam_memory_admit_force_area_ratio": float(getattr(args, "edgetam_memory_admit_force_area_ratio", 0.0)),
                "edgetam_memory_admit_max_masks_per_frame": int(getattr(args, "edgetam_memory_admit_max_masks_per_frame", 0)),
                "edgetam_memory_admit_sort_by": str(getattr(args, "edgetam_memory_admit_sort_by", "area_desc")),
            },
            "v106_runtime_breakdown": _aggregate_frame_diagnostics(summary),
            "edgetam_memory_admission_runtime": get_runtime_stats(),
            "wrapper_wall_time_sec": float(wall_sec),
            "wrapper_wall_time_human": _format_seconds(wall_sec),
        }
    )
    visual = _write_side_by_side_video(
        summary=summary,
        output_root=run_root,
        fps=float(args.fps),
        variant_id=VARIANT_ID,
        layout="RGB frame | v106 SAM2 segmentor + EdgeTAM tracker overlay",
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


if __name__ == "__main__":
    raise SystemExit(main())
