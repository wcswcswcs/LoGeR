from __future__ import annotations

from typing import Any

from .d4rt_scene_builder import D4RTNativeSceneBuilder


def estimate_overlap_self_sim3(
    previous_chunk: dict[str, Any],
    current_chunk: dict[str, Any],
    *,
    min_points: int = 4,
    min_inlier_abs010: float = 0.50,
) -> dict[str, Any] | None:
    builder = D4RTNativeSceneBuilder(
        object(),
        {"model": {"input": {"clip_frames": 32}}},
        temporal_chunk_size=32,
        temporal_chunk_stride=16,
    )
    return builder.estimate_overlap_self_sim3(
        previous_chunk,
        current_chunk,
        min_points=int(min_points),
        min_inlier_abs010=float(min_inlier_abs010),
    )

