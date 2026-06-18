#!/usr/bin/env python3
"""Summarize proposal tracklets stored by video_masklet_front_end_v2 cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize proposal tracklet boxes by frame.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--tracklets", required=True, help="Comma-separated proposal tracklet indices.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    return parser.parse_args()


def as_box(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return []
    return [round(float(x), 1) for x in arr[:4].tolist()]


def get_by_frame(mapping: dict[Any, Any], frame_idx: int) -> Any:
    if int(frame_idx) in mapping:
        return mapping[int(frame_idx)]
    return mapping.get(str(int(frame_idx)))


def frame_keys(tracklet: dict[str, Any]) -> list[int]:
    if "box_by_frame" in tracklet and isinstance(tracklet.get("box_by_frame"), dict):
        return sorted(int(k) for k in tracklet.get("box_by_frame", {}).keys())
    if isinstance(tracklet.get("frames"), (list, tuple)):
        return sorted(int(k) for k in tracklet.get("frames", []))
    return []


def get_box(tracklet: dict[str, Any], frame_idx: int) -> Any:
    mapping = tracklet.get("box_by_frame", {}) if "box_by_frame" in tracklet else None
    if isinstance(mapping, dict):
        return get_by_frame(mapping, frame_idx)
    frames = [int(k) for k in tracklet.get("frames", [])] if isinstance(tracklet.get("frames"), (list, tuple)) else []
    if int(frame_idx) not in frames:
        return None
    pos = frames.index(int(frame_idx))
    boxes = tracklet.get("boxes")
    if boxes is None:
        return None
    return boxes[pos]


def main() -> None:
    args = parse_args()
    payload = torch.load(Path(args.input_pt), map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "tracklets" in payload:
        tracklets = list(payload["tracklets"])
        source_key = "tracklets"
    elif isinstance(payload, dict) and "tracks" in payload:
        tracklets = list(payload["tracks"])
        source_key = "tracks"
    elif isinstance(payload, (list, tuple)):
        tracklets = list(payload)
        source_key = "list"
    else:
        raise SystemExit(f"Unsupported payload structure in {args.input_pt}: {type(payload).__name__}")
    requested = [int(x.strip()) for x in str(args.tracklets).split(",") if x.strip()]
    print(f"input={args.input_pt}")
    print(f"source_key={source_key}")
    print(f"tracklet_count={len(tracklets)}")
    for idx in requested:
        if idx < 0 or idx >= len(tracklets):
            print(f"TRACKLET {idx}: missing")
            continue
        tracklet = tracklets[idx]
        frames = frame_keys(tracklet)
        if not frames:
            print(f"TRACKLET {idx}: empty")
            continue
        label = tracklet.get("label", tracklet.get("L_sem", ""))
        print(
            f"TRACKLET {idx}",
            f"label={label}",
            f"source={tracklet.get('source', tracklet.get('source_type', ''))}",
            f"span={frames[0]}-{frames[-1]}",
            f"visible={len(frames)}",
            f"score={tracklet.get('score', '')}",
        )
        end = int(args.end)
        if end < 0:
            end = frames[-1]
        for frame_idx in range(int(args.start), end + 1):
            box = as_box(get_box(tracklet, frame_idx))
            if box:
                print(f"  {frame_idx}: {box}")


if __name__ == "__main__":
    main()
