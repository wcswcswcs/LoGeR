#!/usr/bin/env python3
"""Run the legacy LSeg STUFF backend as a standalone sparse output.

This tool reuses the LSeg implementation already present in the v1 video
masklet runner, but avoids running detector/SAM stages. It is meant for
auditing LSeg as a conservative stuff backend on the same videos used by v2.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from export_sparse_masklet_slice import _make_single_contact  # noqa: E402
from refine_sparse_stuff_masks import coverage_stats, parse_contact_frames, track_stats  # noqa: E402
from run_video_masklet_front_end import (  # noqa: E402
    SparseMaskletOutput,
    _run_lseg_stuff_inprocess_payload,
    collect_image_paths,
    prepare_processing_image_paths,
)
from run_video_masklet_front_end_v2 import create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSeg STUFF-only sparse output.")
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames_limit", type=int, default=300)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--labels", default="wall,floor,ceiling")
    parser.add_argument(
        "--background_prompts",
        default="person,people,clothing,hair,bag,chair,table,furniture,object,screen,sky,tree,vegetation,other",
    )
    parser.add_argument("--prompt_template", default="there is a {classname} in the scene")
    parser.add_argument("--confidence_threshold", type=float, default=0.15)
    parser.add_argument("--min_area_ratio", type=float, default=0.010)
    parser.add_argument("--max_area_ratio", type=float, default=0.85)
    parser.add_argument("--morph_kernel", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lseg_max_side", type=int, default=512)
    parser.add_argument("--lseg_repo_root", default="third_party/LSM")
    parser.add_argument("--lseg_checkpoint", default="ckpts/LSeg/demo_e200.ckpt")
    parser.add_argument("--lseg_device", default="auto")
    parser.add_argument("--lseg_half_res", type=int, default=0)
    parser.add_argument("--lseg_amp", type=int, default=1)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


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


def _make_lseg_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda" if str(args.lseg_device).strip().lower() == "auto" else str(args.lseg_device),
        efficientsam3_stuff_enable=1,
        efficientsam3_stuff_stride=1,
        lseg_stuff_repo_root=str(args.lseg_repo_root),
        lseg_stuff_checkpoint=str(args.lseg_checkpoint),
        lseg_stuff_prompts=str(args.labels),
        lseg_stuff_prompt_template=str(args.prompt_template),
        lseg_stuff_background_prompts=str(args.background_prompts),
        lseg_stuff_confidence_threshold=float(args.confidence_threshold),
        lseg_stuff_min_area_ratio=float(args.min_area_ratio),
        lseg_stuff_max_area_ratio=float(args.max_area_ratio),
        lseg_stuff_morph_kernel=int(args.morph_kernel),
        lseg_stuff_batch_size=int(args.batch_size),
        lseg_stuff_max_side=int(args.lseg_max_side),
        lseg_stuff_half_res=int(args.lseg_half_res),
        lseg_stuff_amp=int(args.lseg_amp),
        lseg_stuff_device=str(args.lseg_device),
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, temp_dirs, proc_shape = _load_processing_frames(args)
    payload = _run_lseg_stuff_inprocess_payload(image_paths, _make_lseg_args(args))
    sparse = SparseMaskletOutput(
        tracks=list(payload.get("tracks", [])),
        num_masklets=len(payload.get("tracks", [])),
        num_frames=len(image_paths),
        frame_height=int(proc_shape[0]),
        frame_width=int(proc_shape[1]),
        debug={"lseg_stuff_standalone": payload.get("debug", {})},
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
        "lseg_debug": payload.get("debug", {}),
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
