from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ChunkWindow:
    scene_id: str
    chunk_index: int
    start_frame: int
    end_frame_exclusive: int
    overlap: int


def build_sequential_windows(
    scene_id: str, frame_count: int, chunk_size: int, overlap: int
) -> List[ChunkWindow]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    windows: List[ChunkWindow] = []
    start = 0
    index = 0
    step = chunk_size - overlap
    while start < frame_count:
        end = min(frame_count, start + chunk_size)
        windows.append(
            ChunkWindow(
                scene_id=scene_id,
                chunk_index=index,
                start_frame=start,
                end_frame_exclusive=end,
                overlap=overlap if index > 0 else 0,
            )
        )
        if end == frame_count:
            break
        start += step
        index += 1
    return windows


def assert_single_scene_serial_order(windows: Iterable[ChunkWindow]) -> None:
    previous_index = -1
    previous_scene = None
    for window in windows:
        if previous_scene is not None and window.scene_id != previous_scene:
            raise AssertionError("single-scene schedule contains multiple scene ids")
        if window.chunk_index != previous_index + 1:
            raise AssertionError("chunk windows are not in serial chunk-index order")
        previous_index = window.chunk_index
        previous_scene = window.scene_id

