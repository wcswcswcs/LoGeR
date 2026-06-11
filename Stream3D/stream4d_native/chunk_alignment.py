from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChunkPolicy:
    clip_frames: int
    temporal_chunk_size: int
    temporal_chunk_stride: int
    temporal_chunk_overlap: int
    source: str


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def read_checkpoint_clip_frames(checkpoint_config: str | Path | dict[str, Any]) -> int:
    """Read ``model.input.clip_frames`` from an OpenD4RT checkpoint config."""

    if isinstance(checkpoint_config, dict):
        value = _nested_get(checkpoint_config, ("model", "input", "clip_frames"))
        if value is None:
            raise ValueError("checkpoint config missing model.input.clip_frames")
        return max(1, int(value))

    path = Path(checkpoint_config)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml

        data = yaml.safe_load(text)
        value = _nested_get(data, ("model", "input", "clip_frames"))
        if value is not None:
            return max(1, int(value))
    except Exception:
        pass

    # Conservative fallback for environments without PyYAML.
    match = re.search(r"(?m)^\s*clip_frames\s*:\s*(\d+)\s*$", text)
    if not match:
        raise ValueError(f"Could not read clip_frames from {path}")
    return max(1, int(match.group(1)))


def build_checkpoint_chunk_policy(
    checkpoint_config: str | Path | dict[str, Any],
    *,
    temporal_chunk_size: int | None = None,
    temporal_chunk_stride: int | None = None,
    full_scene_method: bool = True,
) -> ChunkPolicy:
    clip_frames = read_checkpoint_clip_frames(checkpoint_config)
    chunk_size = clip_frames if temporal_chunk_size is None else int(temporal_chunk_size)
    stride = max(1, clip_frames // 2) if temporal_chunk_stride is None else int(temporal_chunk_stride)
    if chunk_size > clip_frames:
        raise ValueError(f"temporal_chunk_size={chunk_size} exceeds checkpoint clip_frames={clip_frames}")
    if full_scene_method and stride >= chunk_size:
        raise ValueError("full-scene D4RT-native methods require overlapping chunks: stride < chunk_size")
    overlap = chunk_size - stride
    source = str(checkpoint_config) if not isinstance(checkpoint_config, dict) else "dict"
    return ChunkPolicy(
        clip_frames=int(clip_frames),
        temporal_chunk_size=int(chunk_size),
        temporal_chunk_stride=int(stride),
        temporal_chunk_overlap=int(overlap),
        source=source,
    )


def make_sliding_window_clip_ranges(
    num_frames: int,
    clip_frames: int,
    stride: int | None = None,
) -> list[tuple[int, int]]:
    num_frames = max(0, int(num_frames))
    clip_frames = max(1, int(clip_frames))
    if num_frames == 0:
        return []
    if num_frames <= clip_frames:
        return [(0, num_frames)]
    step = max(1, clip_frames // 2) if stride is None else max(1, int(stride))
    if step >= clip_frames:
        raise ValueError("sliding windows for full-scene D4RT-native methods must overlap")

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        end = min(num_frames, start + clip_frames)
        ranges.append((int(start), int(end)))
        if end >= num_frames:
            break
        start += step
    last_start = max(0, num_frames - clip_frames)
    if ranges[-1][0] != last_start:
        ranges.append((int(last_start), int(num_frames)))
    for begin, end in ranges:
        if end - begin > clip_frames:
            raise AssertionError("internal error: window exceeds checkpoint clip length")
    return ranges
