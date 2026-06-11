from __future__ import annotations

import numpy as np


def make_anchor_clip_indices(num_frames: int, clip_frames: int, target_idx: int, source_idx: int = 0) -> np.ndarray:
    """Port of OpenD4RT ``infer_track_3d._make_anchor_clip_indices``.

    It builds a legal local clip containing both the source frame and target
    frame. The function is used only for sparse long-range tracking diagnostics,
    not for primary scene geometry.
    """

    num_frames = int(num_frames)
    clip_frames = max(1, int(clip_frames))
    if num_frames <= 0:
        return np.empty((0,), dtype=np.int64)
    target_idx = int(np.clip(int(target_idx), 0, max(0, num_frames - 1)))
    source_idx = int(np.clip(int(source_idx), 0, max(0, num_frames - 1)))
    if num_frames <= clip_frames:
        return np.arange(num_frames, dtype=np.int64)
    if clip_frames == 1:
        return np.asarray([target_idx], dtype=np.int64)

    if source_idx != 0:
        window = [int(source_idx)]
        tail_len = clip_frames - 1
        seg_end = target_idx + 1
        seg_start = max(0, seg_end - tail_len)
        seg_end = min(num_frames, seg_start + tail_len)
        seg_start = max(0, seg_end - tail_len)
        for frame_idx in range(seg_start, seg_end):
            if frame_idx != source_idx:
                window.append(int(frame_idx))
        if target_idx not in window:
            window.append(int(target_idx))
        window = sorted(set(window))
        if len(window) > clip_frames:
            mandatory = {int(source_idx), int(target_idx)}
            ranked = sorted(
                [idx for idx in window if idx not in mandatory],
                key=lambda idx: (min(abs(idx - target_idx), abs(idx - source_idx)), idx),
            )
            keep = set(ranked[: max(0, clip_frames - len(mandatory))]) | mandatory
            window = sorted(keep)
        return np.asarray(window, dtype=np.int64)

    tail_len = clip_frames - 1
    seg_end = target_idx + 1
    seg_start = max(1, seg_end - tail_len)
    seg_end = min(num_frames, seg_start + tail_len)
    seg_start = max(1, seg_end - tail_len)
    tail = np.arange(seg_start, seg_end, dtype=np.int64)
    if target_idx not in tail:
        tail[-1] = target_idx
        tail = np.unique(tail)
        while tail.shape[0] < tail_len:
            cand = max(1, int(tail[0]) - 1)
            if cand in tail:
                break
            tail = np.concatenate([np.asarray([cand], dtype=np.int64), tail], axis=0)
        tail = np.sort(tail)[-tail_len:]
    return np.concatenate([np.asarray([0], dtype=np.int64), tail], axis=0)


def validate_anchor_clip_indices(indices: np.ndarray, *, source_idx: int, target_idx: int, clip_frames: int) -> None:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] > int(clip_frames):
        raise ValueError("anchor clip exceeds checkpoint clip_frames")
    if int(source_idx) not in set(indices.tolist()):
        raise ValueError("source frame missing from anchor clip")
    if int(target_idx) not in set(indices.tolist()):
        raise ValueError("target frame missing from anchor clip")
