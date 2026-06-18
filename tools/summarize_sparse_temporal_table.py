#!/usr/bin/env python3
"""Print compact per-frame sparse track activity tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_sparse_stuff_masks import load_sparse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize sparse tracks per frame.")
    parser.add_argument("--input_pt", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--labels", default="all")
    return parser.parse_args()


def label_filter(raw: str) -> set[str] | None:
    text = str(raw).strip()
    if not text or text.lower() in {"all", "*", "off"}:
        return None
    return {part.strip() for part in text.split(",") if part.strip()}


def as_box(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return []
    return [round(float(x), 1) for x in arr[:4].tolist()]


def track_frames(track: dict[str, Any]) -> list[int]:
    return sorted(int(k) for k in track.get("mask_by_frame", {}).keys())


def get_by_frame(mapping: dict[Any, Any], frame_idx: int) -> Any:
    if int(frame_idx) in mapping:
        return mapping[int(frame_idx)]
    return mapping.get(str(int(frame_idx)))


def main() -> None:
    args = parse_args()
    sparse = load_sparse(Path(args.input_pt))
    labels = label_filter(args.labels)
    tracks = list(sparse.tracks)
    print(f"input={args.input_pt}")
    print(f"tracks={len(tracks)} frames={int(sparse.num_frames)} labels={args.labels}")
    for idx, track in enumerate(tracks):
        label = str(track.get("L_sem", ""))
        if labels is not None and label not in labels:
            continue
        frames = track_frames(track)
        if not frames:
            continue
        print(
            "track",
            idx,
            f"label={label}",
            f"source={track.get('source_type', '')}",
            f"status={track.get('refine_status', '')}",
            f"span={frames[0]}-{frames[-1]}",
            f"visible={len(frames)}",
            f"proposal={track.get('proposal_tracklet_id', '')}",
        )
    print("--- per-frame")
    end = int(args.end)
    if end < 0:
        end = int(sparse.num_frames) - 1
    for frame_idx in range(int(args.start), end + 1):
        rows: list[str] = []
        for idx, track in enumerate(tracks):
            label = str(track.get("L_sem", ""))
            if labels is not None and label not in labels:
                continue
            if get_by_frame(track.get("mask_by_frame", {}), frame_idx) is None:
                continue
            box = as_box(get_by_frame(track.get("box_by_frame", {}), frame_idx))
            rows.append(f"{idx}:{label}:{box}")
        print(f"{frame_idx}: " + " | ".join(rows))


if __name__ == "__main__":
    main()
