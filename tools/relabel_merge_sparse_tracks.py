#!/usr/bin/env python3
"""Relabel and merge sparse masklet tracks.

This is an auditable post-processing tool for semantic backends that split one
scene concept across adjacent fixed-vocabulary labels, e.g. VSPW
``curtain``/``textiles`` for the same Taylor stage curtain.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import (  # noqa: E402
    _unpack_mask_np,
    clone_sparse,
    coverage_stats,
    load_sparse,
    make_contact_sheet,
    parse_contact_frames,
    track_stats,
)
from run_video_masklet_front_end import collect_image_paths, prepare_processing_image_paths  # noqa: E402
from run_video_masklet_front_end_v2 import _make_track, _write_mask, create_tracking_video_v2, save_sparse_output  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel and merge sparse masklet tracks.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--relabel",
        required=True,
        help="Comma/semicolon separated src:dst pairs, e.g. 'textiles:curtain;cloth:curtain'.",
    )
    parser.add_argument("--only_source_type", default="stuff_static", help="Only relabel tracks with this source_type, or all.")
    parser.add_argument("--frames_limit", type=int, default=0)
    parser.add_argument("--processing_max_side", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mask_alpha", type=float, default=0.38)
    parser.add_argument("--contact_frames", default="0,30,60,90,120,150,180,210,240,270,299")
    return parser.parse_args()


def _norm(label: Any) -> str:
    return str(label or "").strip().lower().replace(" ", "_")


def _parse_relabel(spec: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw_item in str(spec or "").replace(",", ";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid relabel item {item!r}; expected src:dst")
        src, dst = item.split(":", 1)
        src_norm = _norm(src)
        dst_norm = _norm(dst)
        if not src_norm or not dst_norm:
            raise ValueError(f"Invalid relabel item {item!r}; empty src/dst")
        mapping[src_norm] = dst_norm
    if not mapping:
        raise ValueError("No relabel mapping provided")
    return mapping


def _load_processing_frames(args: argparse.Namespace, expected_h: int, expected_w: int, num_frames: int) -> tuple[List[str], List[str]]:
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
    image_paths = image_paths[:num_frames]
    if tuple(proc_shape) != (expected_h, expected_w):
        raise RuntimeError(f"Frame shape {proc_shape} does not match sparse shape {(expected_h, expected_w)}")
    return list(image_paths), temp_dirs


def _should_relabel(track: Dict[str, Any], mapping: Dict[str, str], only_source_type: str) -> bool:
    if _norm(track.get("L_sem")) not in mapping:
        return False
    source_filter = str(only_source_type or "").strip()
    if not source_filter or source_filter.lower() == "all":
        return True
    return str(track.get("source_type", "")) == source_filter


def _merge_sparse(sparse: Any, mapping: Dict[str, str], only_source_type: str) -> Tuple[Any, Dict[str, Any]]:
    H, W = int(sparse.frame_height), int(sparse.frame_width)
    merged = clone_sparse(sparse)
    output_tracks: List[Dict[str, Any]] = []
    accum: Dict[Tuple[str, str], Dict[int, np.ndarray]] = {}
    birth: Dict[Tuple[str, str], int] = {}
    source_labels: Dict[str, List[str]] = {}
    relabeled_track_count = 0

    destination_labels = set(mapping.values())
    for track in sparse.tracks:
        label_norm = _norm(track.get("L_sem"))
        source_type = str(track.get("source_type", ""))
        eligible_source = str(only_source_type or "").strip().lower() in {"", "all"} or source_type == only_source_type
        participates = eligible_source and (label_norm in mapping or label_norm in destination_labels)
        if not participates:
            output_tracks.append(clone_sparse(type("Tmp", (), {
                "tracks": [track],
                "num_frames": sparse.num_frames,
                "frame_height": H,
                "frame_width": W,
                "debug": {},
                "num_masklets": 1,
            })()).tracks[0])
            continue

        dst_label = mapping.get(label_norm, label_norm)
        key = (source_type, dst_label)
        birth[key] = min(int(birth.get(key, track.get("birth_frame", 0))), int(track.get("birth_frame", 0)))
        source_labels.setdefault(dst_label, [])
        if label_norm not in source_labels[dst_label]:
            source_labels[dst_label].append(label_norm)
        if label_norm in mapping:
            relabeled_track_count += 1
        frame_masks = accum.setdefault(key, {})
        for frame_idx, packed in track.get("mask_by_frame", {}).items():
            frame_idx_int = int(frame_idx)
            mask = _unpack_mask_np(np.asarray(packed, dtype=np.uint8), H, W)
            if frame_idx_int in frame_masks:
                frame_masks[frame_idx_int] |= mask
            else:
                frame_masks[frame_idx_int] = mask.copy()

    for (source_type, dst_label), frame_masks in sorted(accum.items(), key=lambda item: (item[0][0], item[0][1])):
        track = _make_track(dst_label, source_type, int(birth.get((source_type, dst_label), 0)), H, W, "relabel_merge", None)
        for frame_idx in sorted(frame_masks):
            _write_mask(track, int(frame_idx), frame_masks[frame_idx], 1.0, H, W)
        if track.get("mask_by_frame"):
            output_tracks.append(track)

    merged.tracks = output_tracks
    merged.num_masklets = len(output_tracks)
    debug = {
        "mapping": mapping,
        "only_source_type": str(only_source_type),
        "input_tracks": int(len(sparse.tracks)),
        "output_tracks": int(len(output_tracks)),
        "relabeled_track_count": int(relabeled_track_count),
        "merged_destinations": {label: sorted(values) for label, values in sorted(source_labels.items())},
    }
    merged.debug["relabel_merge_sparse_tracks"] = debug
    return merged, debug


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse(Path(args.input_pt))
    mapping = _parse_relabel(args.relabel)
    merged, debug = _merge_sparse(sparse, mapping, str(args.only_source_type))
    image_paths, temp_dirs = _load_processing_frames(args, merged.frame_height, merged.frame_width, merged.num_frames)

    output_pt = output_dir / "sparse_masklets.pt"
    output_video = output_dir / "overlay_final.mp4"
    contact_path = output_dir / "contact_before_after.jpg"
    metrics_path = output_dir / "metrics_summary.json"

    save_sparse_output(output_pt, merged)
    create_tracking_video_v2(
        image_paths,
        merged,
        str(output_video),
        fps=int(args.fps),
        mask_alpha=float(args.mask_alpha),
        render_style="clean",
    )
    make_contact_sheet(
        image_paths,
        sparse,
        merged,
        parse_contact_frames(args.contact_frames, merged.num_frames),
        contact_path,
        float(args.mask_alpha),
    )
    before = coverage_stats(sparse)
    after = coverage_stats(merged)
    summary = {
        "input_pt": str(args.input_pt),
        "output_pt": str(output_pt),
        "output_video": str(output_video),
        "contact_sheet": str(contact_path),
        "before": before,
        "after": after,
        "delta": {key: float(after[key]) - float(before[key]) for key in before if key in after},
        "track_stats_after": track_stats(merged),
        "relabel_merge_debug": debug,
    }
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    for tmp in reversed(temp_dirs):
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
